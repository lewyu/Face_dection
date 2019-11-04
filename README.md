# Face_dection
【Requirement】

模型数据集：链接：https://pan.baidu.com/s/1GMKtsczUkaCJwGTkwYEb9g 提取码：8vsr 

MTCNN：https://github.com/ipazc/mtcnn

FaceNethttps://github.com/davidsandberg/facenet

TensorFlow：1.7 


【Abstract】In recent years, various target detection and recognition technologies have been popularized in many application scenarios, and the main reason is that computer vision continues to make new breakthroughs. Among them, face detection and recognition technology is widely used in security, finance and other industries. In this paper, the development history of face recognition and the status quo at home and abroad are introduced in detail. Finally, the deep learning framework TensorFlow is used in the Python integrated development tool Pycharm. Based on the MTCNN/FaceNet/OpenCV environment, the face recognition and online detection system is designed and implemented. The face online detection module is implemented by the MTCNN model, and then the FaceNet model is used to compare and recognize the detected faces to realize the face recognition function.

【Key words】 Face recognition, Face detection, Deep learning.

1.近年来, 各种目标检测及识别技术在诸多应用场景得到普及，其主要原因在于计算机视觉不断取得新突破。其中作为热点之一，人脸检测及识别技术被广泛应用于安防、金融等行业。本文详细介绍了人脸识别的发展历史和国内外现状，最后在Python集成开发工具Pycharm上，采用深度学习框架TensorFlow，基于MTCNN/FaceNet/OpenCV环境，设计并实现了人脸识别及在线检测系统。其中人脸在线检测模块由MTCNN模型实现，再利用FaceNet模型对检测后的人脸进行对比和识别，实现人脸识别的功能。

2.1 MTCNN模型

MTCNN[10] (,Multi-task convolutional neural network)综合考虑了人脸区域检测和面部关键点检测，称为多任务级联神经网络模型。MTCNN模型的网络整体架构如下图2.1所示：

 ![image](https://github.com/lewyu/Face_dection/blob/master/readme_img/1.jpg)
 
图2.1 MTCNN 整体架构


首先，根据不同的缩放将照片缩放到不同的尺寸，以形成图像的特征金字塔。P-Net通过标记面部区域窗口并获取边界框坐标位置。在得到若干边界框后，然后通过非最大抑制（NMS）合并高度重叠的候选帧。R-Net将通过R-Net网络中的P-Net候选框进行训练，然后使用边界框的回归值来微调候选表格，然后使用NMS删除重叠表格。 O-Net功能类似于R-Net，除了在删除重叠候选窗口的同时显示五个面部关键点。
P-Net（Proposal Network）的网络结构是一个完整的卷积神经网络结构和一个浅网络，因此对输入图片的大小没有要求，但是不同大小的图片会产生不同数量的输出。该网络结构目的是为了得到候选窗口和面部区域的边界框的回归向量。边界框用于回归，并且校准候选窗口。P-Net的模型结构如图2.2所示：

 ![image](https://github.com/lewyu/Face_dection/blob/master/readme_img/2.png)
 
图2.2 P-Net

R-Net(Refine Network)比P-Net多了最后一层全连接层，因此输入固定大小的图片，得到固定数量的输出，但这样一来会取得更好的抑制假积极的效果。这个网络的主要任务是将经过P-Net确定的包含候选框的图像在R-Net网络中继续训练，使用边界框向量微调候选框，然后使用非最大值抑制删除重叠边界以进一步细化面部候选框。R-Net 的模型结构如图2.3所示：

 ![image](https://github.com/lewyu/Face_dection/blob/master/readme_img/3.png)
 
图2.3 R-Net

O-Net（Output Network）是MTCNN中用作网络最终输出的最后一个网络。 该层比R-Net层又多一层卷积层，因此也对输入图片的大小有要求，同时得到固定数量的输出。其作用和R-Net层作用一样，但是该层对面部区域具有更多监督，并且还输出5个界标。处理结果也会进一步精细。O-Net 的模型结构如图2.4所示：

 ![image](https://github.com/lewyu/Face_dection/blob/master/readme_img/4.png)
 
图2.4 O-Net

2.2 FaceNet模型

FaceNet是谷歌(Google)公司提出用于人脸识别、验证、聚类功能的模型。与传统用于分类的神经网络不同，为了使处理过程更高效，FaceNet模型选择了更直接的端到端学习及分类方法。这样一来只会生成更少的参数量，也就提高了处理效率。 执行流程如2.5所示：

 ![image](https://github.com/lewyu/Face_dection/blob/master/readme_img/5.png)
 
图2.5 FaceNet执行流程图

输入 Batch，有一个提取特征的深度神经网络，然后对网络提取到的特征值做L2-normalization，之后通过embedding编码生成128d的向量，优化三元组损失函数得到最优模型。embedding 的数学解释f(x)ϵR^d,d = 128,其中x 表示输入图像，存在恒等式
‖f(x)‖_2=1。

FaceNet是通过学习一个固定为128维度的embedding 空间向量，然后将每一张人脸面部图像都用一个只属于自己的向量表示，再通过计算向量间的距离来实现这一部分的功能。用三重损失(triplets-loss)代替了常用作输出层的归一化指数函数(soft Max)，直接把图像映射到128维的embedding空间然后计算欧式距离，更直接和高效。在当年LFW竞赛中获得了最高分，识别率为99.63％。

使用该模型的主要思想是将面部图像映射到多维空间，并且空间相似性表示面部的相似性。该空间距离称为欧几里德距离。欧氏距离的核心是，假设图像矩阵有 n个元素（n个像素），使用 n个元素值（ x1， x2，...，xn）图像的特征集（像素点矩阵中的所有像素），特征集形成 n维空间（欧洲距离用于多维空间），特征组中的特征代码（每个像素）构成每个维度的值[4]。即x1（第一个像素）对应于一个维度。X2（第二像素）对应于二维，并且xn（第n像素）对应于n维。在n维空间中，两个图像矩阵各自形成一个点，然后通过数学欧几里德距离公式计算两点之间的距离，并且最小距离是最佳匹配图像。欧氏距离计算公式如下：

<a ><img src="https://latex.codecogs.com/gif.latex?dist(x,y)=\sqrt{\sum_{i=1}^{n}\left&space;(&space;x_i{}&space;\right-&space;y_i{}&space;)^{2}}" title="dist(x,y)=\sqrt{\sum_{i=1}^{n}\left ( x_i{} \right- y_i{} )^{2}}" /></a>


不同人的面部图像计算得到的的欧式距离比较大，而同一个人的不同面部图像的欧式距离比较小。通过比较不同面部图片之间的欧式距离，就可以判定两张图片是否属于同一个人，当空间距离为0时，表示为同一张图片。由此将人脸面部图像投影到多维空间来实现人脸识别的过程。

2.3 LFW人脸数据集

LFW（Labeled Faces in the Wild）人脸面部数据集是由马萨诸塞州立大学阿默斯特分校的计算机视觉实验室编制的数据库。LFW人脸数据库是无约束自然场景人脸识别数据集，用于研究并解决非实验室环境下下的人脸识别问题。主要从互联网资源而不是实验室收集图像。
它由13000多张全球各地名人在不同场合、不同面部情绪、不同光线条件下的人脸照片组成，共有五千多人。其中，1680人对应多个图像，即约1680人包含两个以上的面部图像。每个人脸图像都有其唯一的名字和编号来区别。通过6000对人脸测试结果的系统答案与真实答案的比值可以得到人脸识别准确率。该数据集广泛用于评估面部验证算法的性能。

