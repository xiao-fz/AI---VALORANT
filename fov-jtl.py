import time
import math
from ultralytics import YOLO
import ctypes
import ctypes.wintypes
import os
from ctypes import *
import rapidshot
import torch
import torch.nn.functional as F

# ==================== 配置区（只改这里） ====================
# 按键
PAUSE_KEY = 0x54            # T 键，暂停/继续
PAUSE_SLEEP_S = 0.01        # 暂停时循环间隔（秒）

# 鼠标 / 瞄准
GAME_SENSITIVITY = 5        # 游戏内鼠标灵敏度
HORIZONTAL_FOV = 103.0      # 游戏水平 FOV
VERTICAL_FOV = 70.53        # 游戏垂直 FOV
M_YAW = 0.018               # 游戏 m_yaw，一般不用改
MAX_STEP = 100              # 预留步长上限（实际移动按 DD -127~126 切）
MULTIPLIER_SCALE = 0.71     # 瞄准倍率微调，越大动得越猛

# 截图 / 模型
CAPTURE_SIZE = 320          # 屏幕中心截图边长（像素）
MODEL_IMGSZ = 640           # YOLO 输入尺寸，必须和 engine 一致
CONF_THRES = 0.3            # YOLO 置信度阈值
MODEL_PATH = "best.engine"  # 模型文件
DD_DLL_PATH = "ddhid60400.dll"  # DD 驱动文件
CAPTURE_TIMEOUT_MS = 0      # RapidShot 等待新帧（毫秒），0=空轮询
GRAB_WAIT_MS = 1            # 空轮询没帧时再等的毫秒
# ==================== 配置区结束 ====================

NVIDIA_VENDOR_ID = 0x10DE
_GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState


def _ensure_cupy_cuda_dlls():
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib):
        return
    os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib)
        except OSError:
            pass


def auto_select_best_cuda_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_device = 0
    best_memory = 0
    for i in range(torch.cuda.device_count()):
        memory = torch.cuda.get_device_properties(i).total_memory
        if memory > best_memory:
            best_memory = memory
            best_device = i
    return torch.device(f"cuda:{best_device}")


def _cupy_usable():
    if not torch.cuda.is_available():
        return False
    try:
        import cupy as cp
        x = cp.zeros((1,), dtype=cp.uint8)
        del x
        return True
    except Exception:
        return False


def auto_create_capture(region, timeout_ms=CAPTURE_TIMEOUT_MS, verbose=False):
    nvidia_gpu = _cupy_usable()
    rapidshot.reset()
    factory = rapidshot.get_factory()
    if not factory.devices:
        raise RuntimeError("没有可截图的显卡（没有接到显示器的适配器）")
    if verbose:
        print(rapidshot.topology_info())

    candidates = []
    for didx, device in enumerate(factory.devices):
        desc = device.description or ""
        low = desc.lower()
        if "microsoft basic" in low or "basic render" in low:
            continue
        vendor = int(getattr(device, "vendor_id", 0) or 0)
        is_nvidia = vendor == NVIDIA_VENDOR_ID or "nvidia" in low
        for oidx, output in enumerate(factory.outputs[didx]):
            meta = factory.output_metadata.get(output.devicename)
            is_primary = bool(meta and len(meta) > 1 and meta[1])
            candidates.append((
                0 if is_primary else 1,
                0 if is_nvidia else 1,
                didx, oidx, desc, is_primary,
            ))
    if not candidates:
        for didx, device in enumerate(factory.devices):
            for oidx in range(len(factory.outputs[didx])):
                candidates.append((1, 1, didx, oidx, device.description, False))
    candidates.sort()

    last_err = None
    for use_gpu in ((True, False) if nvidia_gpu else (False,)):
        for _, _, didx, oidx, desc, is_primary in candidates:
            try:
                cam = rapidshot.create(
                    device_idx=didx,
                    output_idx=oidx,
                    region=region,
                    output_color="BGRA",
                    nvidia_gpu=use_gpu,
                    max_buffer_len=2,
                    timeout_ms=timeout_ms,
                )
                try:
                    frame = cam.grab()
                    if frame is not None and hasattr(frame, "release"):
                        frame.release()
                except Exception:
                    pass
                return cam, {
                    "device_idx": didx,
                    "output_idx": oidx,
                    "desc": desc,
                    "nvidia_gpu": use_gpu,
                    "primary": is_primary,
                }
            except Exception as e:
                last_err = e
                try:
                    rapidshot.reset()
                except Exception:
                    pass
    raise RuntimeError(f"无法创建截图: {last_err}")


def is_key_pressed(key_code):
    return _GetAsyncKeyState(key_code) & 0x8000 != 0


class DDMouseController:
    def __init__(self, dd_dll, screen_width, screen_height):
        self.dd = dd_dll
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_center_x = screen_width // 2
        self.screen_center_y = screen_height // 2
        self.game_sensitivity = None
        self.horizontal_fov = None
        self.vertical_fov = None
        self.m_yaw = M_YAW
        self.max_step = MAX_STEP
        self.game_multiplier = 1.0
        self.dd_step_max = 126
        self.dd_step_min = -127

    def configure(self, game_sensitivity, horizontal_fov, vertical_fov, m_yaw=M_YAW, max_step=MAX_STEP):
        self.game_sensitivity = game_sensitivity
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.m_yaw = m_yaw
        self.max_step = max_step
        self.game_multiplier = self._calculate_multiplier()
        print(f"配置完成: 灵敏度={game_sensitivity}, FOV={horizontal_fov}°, 步长={max_step}, 倍率={self.game_multiplier:.2f}")

    def _calculate_multiplier(self):
        angle_per_pixel_h_deg = math.degrees(math.radians(self.horizontal_fov) / self.screen_width)
        counts_per_pixel = angle_per_pixel_h_deg / self.m_yaw
        return max(0.1, min(100, counts_per_pixel * self.game_sensitivity))

    def set_multiplier(self, multiplier):
        self.game_multiplier = multiplier

    def _dd_movr(self, dx, dy):
        while dx or dy:
            step_x = max(self.dd_step_min, min(self.dd_step_max, dx))
            step_y = max(self.dd_step_min, min(self.dd_step_max, dy))
            self.dd.DD_movR(step_x, step_y)
            dx -= step_x
            dy -= step_y

    def move_to_target(self, target_x, target_y):
        dx_dd = int((target_x - self.screen_center_x) * self.game_multiplier)
        dy_dd = int((target_y - self.screen_center_y) * self.game_multiplier)
        self._dd_movr(dx_dd, dy_dd)

    def reset_to_center(self):
        point = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
        dx_dd = int((self.screen_center_x - point.x) * self.game_multiplier)
        dy_dd = int((self.screen_center_y - point.y) * self.game_multiplier)
        self._dd_movr(dx_dd, dy_dd)


def init_dd(dll_path):
    dll_path = os.path.abspath(dll_path)
    print(f"加载DD驱动: {dll_path}")
    if not os.path.exists(dll_path):
        print(f"错误：找不到DD驱动文件 {dll_path}")
        return None
    try:
        dd_dll = WinDLL(dll_path)
        dd_dll.DD_btn.argtypes = [c_int]
        dd_dll.DD_btn.restype = c_int
        dd_dll.DD_movR.argtypes = [c_int, c_int]
        dd_dll.DD_movR.restype = c_int
        dd_dll.DD_key.argtypes = [c_int, c_int]
        dd_dll.DD_key.restype = c_int
        time.sleep(1)
        if dd_dll.DD_btn(0) == 1:
            print("DD驱动初始化成功")
            return dd_dll
        print("DD驱动初始化失败")
        return None
    except Exception as e:
        print(f"加载DD驱动时出错: {e}")
        return None


def engine_input_dtype(model):
    backend = getattr(getattr(model, "predictor", None), "model", None)
    if backend is not None and getattr(backend, "fp16", False):
        return torch.float16
    return torch.float32


def engine_backend(model):
    return getattr(getattr(model, "predictor", None), "model", None)


def bgra_to_yolo_tensor(gpu_frame, dtype, device, imgsz=MODEL_IMGSZ):
    tensor = torch.from_dlpack(gpu_frame) if hasattr(gpu_frame, "__dlpack__") else torch.as_tensor(gpu_frame)
    if tensor.device != device:
        tensor = tensor.to(device, non_blocking=True)
    out = tensor[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
    out.mul_(1.0 / 255.0)
    h, w = out.shape[-2], out.shape[-1]
    coord_scale = 1.0
    if imgsz is not None and (h != imgsz or w != imgsz):
        coord_scale = h / float(imgsz)
        out = F.interpolate(out, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    return out.contiguous(), coord_scale


def closest_screen_target(pred, x_start, y_start, screen_center_x, screen_center_y,
                          coord_scale=1.0, conf_thres=CONF_THRES):
    if pred is None:
        return None, None
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    boxes = pred[0] if pred.ndim == 3 else pred
    if boxes is None or boxes.numel() == 0:
        return None, None
    boxes = boxes[boxes[:, 4] >= conf_thres]
    if boxes.shape[0] == 0:
        return None, None
    if boxes.dtype != torch.float32:
        boxes = boxes.float()
    half_s = 0.5 * coord_scale
    px = (boxes[:, 0] + boxes[:, 2]) * half_s + x_start
    py = (boxes[:, 1] + boxes[:, 3]) * half_s + y_start
    i = torch.argmin((px - screen_center_x) ** 2 + (py - screen_center_y) ** 2)
    tx, ty = torch.stack((px[i], py[i])).tolist()
    return int(tx), int(ty)


def _grab_once(cam, wait_ms=GRAB_WAIT_MS):
    saved = cam.timeout_ms
    try:
        if saved != 0:
            cam.timeout_ms = 0
        frame = cam.grab()
        if frame is not None:
            return frame
        if wait_ms > 0:
            cam.timeout_ms = wait_ms
            return cam.grab()
        return None
    finally:
        if cam.timeout_ms != saved:
            cam.timeout_ms = saved


def run_engine(backend, infer_input):
    if getattr(backend, "engine", False):
        backend.binding_addrs["images"] = int(infer_input.data_ptr())
        backend.context.execute_v2(list(backend.binding_addrs.values()))
        names = backend.output_names
        if len(names) == 1:
            return backend.bindings[names[0]].data
        return [backend.bindings[n].data for n in names]
    with torch.inference_mode():
        return backend(infer_input)


def main():
    dd = init_dd(DD_DLL_PATH)
    if not dd:
        print("无法初始化DD驱动，程序退出")
        return

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except AttributeError:
        ctypes.windll.user32.SetProcessDPIAware()

    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    mouse_controller = DDMouseController(dd, screen_width, screen_height)
    mouse_controller.configure(
        game_sensitivity=GAME_SENSITIVITY,
        horizontal_fov=HORIZONTAL_FOV,
        vertical_fov=VERTICAL_FOV,
        m_yaw=M_YAW,
        max_step=MAX_STEP,
    )
    mouse_controller.set_multiplier(mouse_controller.game_multiplier * MULTIPLIER_SCALE)

    half = CAPTURE_SIZE // 2
    x_start = screen_width // 2 - half
    y_start = screen_height // 2 - half
    region = (x_start, y_start, x_start + CAPTURE_SIZE, y_start + CAPTURE_SIZE)

    _ensure_cupy_cuda_dlls()
    try:
        cam, cap_info = auto_create_capture(region, timeout_ms=CAPTURE_TIMEOUT_MS, verbose=True)
    except Exception as e:
        print(f"截图初始化失败: {e}")
        return

    print(
        f"RapidShot grab 循环已启动（无视频线程）: {CAPTURE_SIZE}x{CAPTURE_SIZE} "
        f"device={cap_info['device_idx']}/{cap_info['desc']} "
        f"output={cap_info['output_idx']} nvidia_gpu={cap_info['nvidia_gpu']} "
        f"backend={cam._processor.active_backend.name}"
    )

    try:
        device = auto_select_best_cuda_device()
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model = YOLO(MODEL_PATH, task="detect")
        warmup = torch.zeros(1, 3, MODEL_IMGSZ, MODEL_IMGSZ, dtype=torch.float16, device=device)
        model(warmup, conf=CONF_THRES, verbose=False, device=device, half=True)
        backend = engine_backend(model)
        dtype = engine_input_dtype(model)
        if backend is None:
            raise RuntimeError("YOLO predictor 未初始化")
        precision = "fp16" if dtype == torch.float16 else "fp32"
        print(
            f"YOLO模型加载成功，捕获 {CAPTURE_SIZE} → 推理 {MODEL_IMGSZ}，"
            f"推理设备 {device}，engine={precision}"
        )
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        cam.release()
        return

    pause_flag = False
    last_t_state = False
    prev_move_x = 0
    prev_move_y = 0

    print("脚本已启动！grab 循环伪装视频流，按T键暂停/继续")

    try:
        mouse_controller.reset_to_center()

        while True:
            t_pressed = is_key_pressed(PAUSE_KEY)
            if t_pressed and not last_t_state:
                pause_flag = not pause_flag
                if pause_flag:
                    print("\n[暂停] 已暂停 - 按T键继续")
                else:
                    print("\n[继续] 已继续 - 按T键暂停")
            last_t_state = t_pressed
            if pause_flag:
                time.sleep(PAUSE_SLEEP_S)
                continue

            frame = _grab_once(cam)
            if frame is None:
                continue

            try:
                infer_input, coord_scale = bgra_to_yolo_tensor(
                    getattr(frame, "array", frame), dtype, device, imgsz=MODEL_IMGSZ
                )
            finally:
                if hasattr(frame, "release"):
                    frame.release()

            pred = run_engine(backend, infer_input)
            tx, ty = closest_screen_target(
                pred, x_start, y_start,
                mouse_controller.screen_center_x, mouse_controller.screen_center_y,
                coord_scale=coord_scale,
            )

            if tx is not None:
                move_x = tx - mouse_controller.screen_center_x
                move_y = ty - mouse_controller.screen_center_y
                if move_x != prev_move_x or move_y != prev_move_y:
                    mouse_controller.move_to_target(tx, ty)
                prev_move_x, prev_move_y = move_x, move_y

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam.release()
        print("脚本已停止")


if __name__ == "__main__":
    main()
