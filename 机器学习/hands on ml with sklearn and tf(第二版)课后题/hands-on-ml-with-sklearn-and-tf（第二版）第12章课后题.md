---
hands-on-ml-with-sklearn-and-tf-第12章课后题
---





1.How would you describe TensorFlow in a short sentence? What are its main features? Can you name other popular Deep Learning libraries?

```
TensorFlow是一个强大的数值计算库，特别适合做和微调大规模机器学习（但也可以用来做其它的重型计算）。
特点：
1.TensorFlow的核心与NumPy很像，但TensorFlow支持GPU；
2.TensorFlow支持（多设备和服务器）分布式计算；
3.TensorFlow使用了即时JIT编译器对计算速度和内存使用优化。编译器的工作是从Python函数提取出计算图，然后对计算图优化（比如剪切无用的节点），最后高效运行（比如自动并行运行独立任务）；
4.计算图可以导出为迁移形式，因此可以在一个环境中训练一个TensorFlow模型（比如使用Python或Linux），然后在另一个环境中运行（比如在安卓设备上用Java运行）；
5.TensorFlow实现了自动微分，并提供了一些高效的优化器，比如RMSProp和NAdam，因此可以容易的最小化各种损失函数。
```


2.Is TensorFlow a drop-in replacement for NumPy? What are the main differences between the two?

```
相同点：都提供n维数组
不同点：numpy里有ndarray，而Tensorflow里有tensor；numpy不提供创建张量函数和求导，也不提供GPU支持。
```


3.Do you get the same result with tf.range(10) and tf.constant(np.arange(10))?

```
tf.range(10)
<tf.Tensor: id=3, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>

tf.constant(np.arange(10))
<tf.Tensor: id=5, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>
```


4.Can you name six other data structures available in TensorFlow, beyond regular tensors?

```
稀疏张量、张量数组、嵌套张量、字符串张量、集合、队列
```


5.A custom loss function can be defined by writing a function or by subclassing the keras.losses.Loss class. When would you use each option?

```
训练集有噪音。可以通过清除或修正异常值来清理数据集，但是这样还不够：数据集还是有噪音。
均方差可能对大误差惩罚过重，导致模型不准确。
均绝对值误差不会对异常值惩罚过重，但训练可能要比较长的时间才能收敛，训练模型也可能不准确。
```


6.Similarly, a custom metric can be defined in a function or a subclass of keras.metrics.Metric. When would you use each option?

```
创建流式指标，可以创建一个keras.metrics.Metric类的子类。
```

8.What are some use cases that require writing your own custom training loop?

```
当fit()方法不够灵活的时候就需要了。
例如，https://arxiv.org/abs/1606.07792  这篇Wide&Deep论文。
```


9.Can custom Keras components contain arbitrary Python code, or must they be convertible to TF Functions?

```
当你写的自定义损失函数、自定义指标、自定义层或任何其它自定义函数，并在Keras模型中使用的，Keras都自动将其转换成了TF函数，不用使用tf.function()。
创建自定义层或模型时，设置dynamic=True，可以让Keras不转化你的Python函数。
```


10.What are the main rules to respect if you want a function to be convertible to a TF Function?

```
将Python函数转换为TF函数是琐碎的：要用@tf.function装饰，或让Keras来负责。但是，也有一些规则：

如果调用任何外部库，包括NumPy，甚至是标准库，调用只会在跟踪中运行，不会是图的一部分。事实上，TensorFlow图只能包括TensorFlow的构件（张量、运算、变量、数据集，等等）。
因此，要确保使用的是tf.reduce_sum()而不是np.sum()，使用的是tf.sort()而不是内置的sorted()，等等。还要注意：

如果定义了一个TF函数f(x)，它只返回np.random.rand()，当函数被追踪时，生成的是个随机数，因此f(tf.constant(2.))和f(tf.constant(3.))会返回同样的随机数，但f(tf.constant([2., 3.]))会返回不同的数。
如果将np.random.rand()替换为tf.random.uniform([])，每次调用都会返回新的随机数，因为运算是图的一部分。

如果你的非TensorFlow代码有副作用（比如日志，或更新Python计数器），则TF函数被调用时，副作用不一定发生，因为只有函数被追踪时才有效。

你可以在tf.py_function()运算中包装任意的Python代码，但这么做的话会使性能下降，因为TensorFlow不能做任何图优化。还会破坏移植性，因为图只能在有Python的平台上跑起来（且安装上正确的库）。

你可以调用其它Python函数或TF函数，但是它们要遵守相同的规则，因为TensorFlow会在计算图中记录它们的运算。注意，其它函数不需要用@tf.function装饰。

如果函数创建了一个TensorFlow变量（或任意其它静态TensorFlow对象，比如数据集或队列），它必须在第一次被调用时创建TF函数，否则会导致异常。通常，最好在TF函数的外部创建变量（比如在自定义层的build()方法中）。
如果你想将一个新值赋值给变量，要确保调用它的assign()方法，而不是使用=。

Python的源码可以被TensorFlow使用。如果源码用不了（比如，如果是在Python shell中定义函数，源码就访问不了，或者部署的是编译文件*.pyc），图的生成就会失败或者缺失功能。

TensorFlow只能捕获迭代张量或数据集的for循环。因此要确保使用for i in tf.range(x)，而不是for i in range(x)，否则循环不能在图中捕获，而是在会在追踪中运行。（如果for循环使用创建计算图的，这可能是你想要的，比如创建神经网络中的每一层）。

出于性能原因，最好使用矢量化的实现方式，而不是使用循环。
```

