# TomotoHighPredict
随着科技的进步和社会的发展，农业正在经历一场数字化、信息化的革新。智慧农业、精准农业等概念已经广泛渗透。其中深度学习作为人工智能的重要分支，正在逐步应用到农业生产的各个环节。本项目计划利用深度学习的方法，通过输入环境湿度、环境温度、植株照片、施肥情况等参数，预测番茄植株的生长高度。
# 网络结构
![image](https://github.com/Marce-ts/TomotoHighPredict/assets/80326340/64cad660-9f5e-43c7-8f42-2b41ed05f299)
# 部分结果展示
![image](https://github.com/Marce-ts/TomotoHighPredict/assets/80326340/670b5eb9-e84b-46e3-a785-367bf9578ca7)
![image](https://github.com/Marce-ts/TomotoHighPredict/assets/80326340/358e59c9-e2ab-4723-8b7a-049726addfe7)
![image](https://github.com/Marce-ts/TomotoHighPredict/assets/80326340/387aa223-7362-407d-9e5f-1d9215a50836)
# 如何测试
- 图片训练代码读入图片数据，输出模型（pic_model.pth）
- 图片特征提取代码读入模型文件，输出图片特征（excel形式），和传感器和测量数据拼接后成为最终数据（final_data）
- data_make模块读入final_data，划分为组，并将这些组打乱后保存下来（labels_vector_14.npy、data_vector_14.npy、raw_vector_1.npy）
- 训练代码读入data_make保存的数据，输出模型文件(model_vector14_1.npy)，测试预测效果代码在训练代码的注释中。

# 数据下载
- 图片数据下载链接http://47.94.139.248:24193/down/79Dgkl4nFbnT.zip
- 图片特征提取模型http://47.94.139.248:24193/down/Qe2NDoD4QYKD.pth
