b站演示视频：https://www.bilibili.com/video/BV12G8z6aEZd

AI辅助代码（内带瓦的pt模型），需自配置环境（不会弄的可以找ai给你配置，用CC-Switch和下面免费的API）

发布这个代码的初衷是想遇到炸鱼的时候能有一战之力能保留自己的一点游戏体验,

我知道发布这个代码之后肯定会有人骂,手上有挂怎么可能不开呢,是这样的但你说你的但我不听

不是广（纯自推，有grok-4.6）

自用ai中转站推荐：（注册给1$，关注公众号5$，进群更多福利不定时发免费api）

https://ai.yosakurasec.com/pricing

今日免费grok heavy

api： sk-rkRibmmwJH1XskBcdWhX6UGxOCR9vBXoCuTYb9EnP12DgQ

ai挂流程： 获取屏幕 -> 识别人物位置（坐标） -> 移动鼠标

获取屏幕用的：rapidshot 数据不出gpu用了CUDA，但笔记本还是会出gpu的不能完全cuda

识别人物位置：通过训练yolo26n得到的 best.pt（需自转trt）

移动鼠标：至少驱动级或鼠标盒子，我找到的只有dd驱动和IbInputSimulator都是不错的项目

代码直接调用了dd（需自安装），没通过IbInputSimulator再去调用dd
