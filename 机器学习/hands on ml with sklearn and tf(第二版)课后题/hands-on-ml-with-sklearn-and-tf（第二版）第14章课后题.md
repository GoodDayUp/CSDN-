---
hands-on-ml-with-sklearn-and-tf-第14章课后题
---



1.What are the advantages of a CNN over a fully connected DNN for image classification?

```
1.因为连续层只是部分连接而且因为它大量重用其权重，所以CNN比完全连接的DNN具有更少的参数，
这使得训练更快，降低过度拟合的风险，并且需要更少的训练数据。

2.当CNN学习了可以检测特定特征的内核时，它可以在图像上的任何位置检测到该特征。
相反，当DNN在一个位置学习一个特征时，它只能在该特定位置检测到它。
由于图像通常具有非常重复的特征，因此使用较少的训练示例，CNN能够比DNN更好地用于图像处理任务（例如分类）。

3.最后，DNN不知道像素的组织方式;它不知道附近的像素是否接近。 
CNN的架构嵌入了这一先验知识。较低层通常识别图像的小区域中的特征，而较高层将较低层特征组合成较大特征。
这适用于大多数自然图像，与DNN相比，CNN具有决定性的先机性。
```


2.Consider a CNN composed of three convolutional layers, each with 3 × 3 kernels, a stride of 2, and "same" padding. The lowest layer outputs 100 feature maps, the middle one outputs 200, and the top one outputs 400. The input images are RGB images of 200 × 300 pixels.
What is the total number of parameters in the CNN? 

```
(1). 由于第一个卷积层有3×3个内核，输入有三个通道（红色，绿色和蓝色）, 然后每个特征图具有3×3×3权重，加上偏差项。
这是每个特征图的 3×3×3+1= 28个参数。由于该第一卷积层具有100个特征图，因此它具有总共28 × 100 = 2,800个参数。

(2). 第二卷积层具有3×3个核，其输入是前一层的100个特征图的集合，因此每个特征图具有3×3×100 = 900个权重，加上偏差项。
这是每个特征图的 3×3×100 + 1 = 901个参数。 由于第二卷积层有200个特征图，因此该层具有901×200 = 180,200个参数。

(3). 最后，第三个和最后一个卷积层也有3×3个核，其输入是前一个层的200个特征图的集合，因此每个特征图具有3×3×200 = 1,800个权重，加上一个偏置项。
这是每个特征图的 3×3×200+1 = 1801 个参数 由于它有400个特征图，因此该图层总共有1,801×400 = 720,400个参数。

总而言之，CNN有 2,800 + 180,200 + 720,400 = 903,400个参数。
```

If we are using 32-bit floats, at least how much RAM will this network require when making a prediction for a single instance? 
```
(1). 首先让我们计算每一层的特征图大小。由于我们使用步幅2和SAME填充，因此特征图的水平和垂直尺寸在每一层被除以2（必要时向上舍入），
因此输入通道为200×300像素，第一层的特征地图是100×150，第二层的特征地图是50×75，第三层的特征地图是25×38。

(2). 由于32位是4字节而第一卷积层有100个特征图，因此第一层占用4 x 100×150×100 = 600万字节（约5.7 MB，考虑到1 MB = 1,024 KB和1 KB = 1,024字节）。

(3). 第二层占用4×50×75×200 = 300万字节（约2.9MB）。
(4). 第三层占用4×25×38×400 = 1,52万字节（约1.4MB）。

然而，一旦计算了一个层，就可以释放前一层所占用的内存，所以如果一切都经过优化，只需要6 + 9 = 1500万字节（约14.3 MB）的RAM（刚刚计算出第二层时，第一层占用的内存尚未释放）。

但是等等，你还需要添加CNN参数占用的内存。我们之前计算过它有903,400个参数，每个参数使用4个字节，所以这增加了4 × 903,400 = 3,613,600个字节（大约3.4 MB）。

所需的总RAM是（至少）18,613,600字节（约17.8 MB）。
```

What about when training on a mini-batch of 50 images?

```
在训练期间，TensorFlow使用反向传播，这需要保持在前向传递期间计算的所有值，直到反向传递开始。
因此，我们必须计算单个实例的所有层所需的总RAM，并将其乘以50！让我们开始以兆字节而不是字节计数。
我们之前计算过，每个实例的三层分别需要5.7, 2.9 和1.4 MB。

每个实例总共10.0 MB。因此，对于50个实例，总RAM为500 MB。
再加上输入图像所需的RAM，即50×4×200×300×3 = 3600万字节（约34.3 MB），加上模型参数所需的RAM，大约3.4 MB（之前计算过） ，加上梯度的一些RAM。

我们总共大约500.0 + 34.3 +3.4 = 537.7 MB。这真的是一个乐观的最低限度。
```


3.If your GPU runs out of memory while training a CNN, what are five things you could try to solve the problem?

```
1.减少小批量。
2.在一个或多个图层中使用更大的步幅减少维度。
3.删除一个或多个图层。
4.使用16位浮点数而不是32位浮点数。
5.在多个设备上分配CNN。
```


4.Why would you want to add a max pooling layer rather than a convolutional layer with the same stride?

```
最大池层根本没有参数，而卷积层有很多参数。
```


5.When would you want to add a local response normalization layer?

```
局部响应标准化层使最强激活的神经元在相同位置但在相邻特征图中抑制神经元，它鼓励不同的特征图专门化并将它们分开，迫使它们探索更广泛的特征。
它通常在较低层中使用，以具有较大的低级特征池，上层可以构建在其上。
```


6.Can you name the main innovations in AlexNet, compared to LeNet-5? What about the main innovations in GoogLeNet, ResNet, SENet, and Xception?

```
与LeNet-5相比，AlexNet的主要创新是：（1）它更大更深，（2）它将卷积层直接叠加在一起，而不是在每个卷积层的顶部堆叠池化层。
GoogLeNet的主要创新是引入了初始模块，这使得有可能拥有比以前的CNN架构更深的网络，参数更少。
ResNet的主要创新是引入了跳跃连接，这使得它可以超越100层。 可以说，它的简单性和一致性也相当具有创新性。
SENet的主要创新思想是在起始网络中的每个起始模块或ResNet中的每个残差单元之后使用SE块（两层密集网络），以重新校准要素图的相对重要性。
Xception的主要创新之处在于使用了深度可分离的卷积层，它们分别查看了空间模式和深度模式。
```


7.What is a fully convolutional network? How can you convert a dense layer into a convolutional layer?

```
全卷积网络是仅由卷积和池化层组成的神经网络。

FCN可以有效处理任何宽度和高度（至少大于最小尺寸）的图像。
它们对于对象检测和语义分割最有用，因为它们只需要查看一次图像（而不必在图像的不同部分上多次运行CNN）。

如果您的CNN顶部有一些密集层，则可以将这些密集层转换为卷积层以创建FCN：
只需用内核大小等于该层输入大小的卷积层替换最低的密集层，并在密集层中为每个神经元使用一个过滤器，然后使用“有效”填充即可。
通常，步幅应为1，但您可以根据需要将其设置为更高的值。
激活功能应与密集层的相同。

其他密集层应以相同的方式进行转换，但使用1×1滤镜。
实际上，可以通过适当地重塑密集层的权重矩阵来以这种方式转换经过训练的CNN。
```


8.What is the main technical difficulty of semantic segmentation?

```
语义分割的主要技术难题是，随着信号流过每一层，尤其是在合并的层和跨度大于1的层中，CNN会丢失大量空间信息。
需要以某种方式恢复此空间信息，以准确预测每个像素的类别。
```

