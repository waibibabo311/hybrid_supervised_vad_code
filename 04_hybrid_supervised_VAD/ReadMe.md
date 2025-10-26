**MNAD_Ubnormal_Super_20240209**
更改了预训练的2d编码器
使用是Super2Unsuper_FeatureMse_20240205_3那一版的预训练2d编码器

**MNAD_Ubnormal_Super_20240214**
在更多通道的特征级别训练baseline
记忆项数增加至100
学习率降低10倍，降至2e-5
对预训练得到MNAD的特征进行layernorm
去除编码解码器中的batch_norm（待完成）
修改编码解码器层数及通道数（待完成）



**MNAD_Ubnormal_Super_20240311**

1.加入目标检测模块

2.2d特征预提取使用ReconstructionRegionWithBackground_Ablation3_20240229得到的特征（尺寸为512,16,16）



**MNAD_Ubnormal_Super_20240408**

输入特征尺寸（512,32,32） 预特征提取使用ReconstructionRegionWithBackground_Ablation3_20240410
没有检测到异常区域就输入两个原始帧， 检测到异常区域就输入异常区域和原始帧



**MNAD_Ubnormal_Super_20240417**

使用预训练的swin_T模型



**MNAD_Ubnormal_Super_20240515**

使用全连接网络重构



**MNAD_Ubnormal_Super_20240518**

重新跑下MNAD_Ubnormal_Super_20240515，再验证下