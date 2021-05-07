# Face-Parsing-pytorch-opencv-onnxruntime
使用BiSeNet做人脸面部解析，包含了基于pytorch, opencv, onnxruntime三种库的程序实现，并且比较了在调用三种库的输入和输出的差异
.pth文件和.onnx文件从百度云盘下载，
链接：https://pan.baidu.com/s/1VGm7wsfCMw_RH7V_3ODuhg 
提取码：fza0 


基于pytorch框架运行的主程序是main_pytorch.py， 基于opencv运行的是main_opencv.py， 基于onnxruntime运行的是main_onnxrun.py
。在运行程序时，会保存神经网络的输入和输出到.npy文件。运行完这3个程序后，运行cmp_debug.py，它会比较在调用这三个不同框架时，
同一个神经网络的输入和输出的差异。

BiSeNet是一个语义分割网络，人脸面部解析的本质是对人脸的不同器官做分割或者说像素级分类。本程序里，在运行cmp_debug.py后发现，调用
pytorch框架的输出和调用opencv和onnxruntime的输出都不同，而opencv和onnxruntime的输出差异仅仅在小数点后10位，可以认为两者相等。
那么究竟是什么原因导致调用opencv或者onnxruntime的输出与调用pytorch的输出不同呢？
从运行程序的可视化结果图看，调用pytorch库的程序的输出结果是正确的，转换生成onnx文件的程序在net.py里，读者可以继续调试排查原因
