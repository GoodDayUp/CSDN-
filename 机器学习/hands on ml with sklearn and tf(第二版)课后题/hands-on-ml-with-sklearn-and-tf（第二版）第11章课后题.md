---
hands-on-ml-with-sklearn-and-tf-第11章课后题
---





1.Is it OK to initialize all the weights to the same value as long as that value is selected randomly using He initialization?

```
不，所有权重都应独立采样;它们不应该都具有相同的初始值。随机抽样权重的一个重要目的是打破对称性：
如果所有权重具有相同的初始值，即使该值不为零，那么对称性不会被破坏（即，给定层中的所有神经元都是等价的），并且反向传播也将无法打破它。

具体地说，这意味着任何给定层中的所有神经元将始终具有相同的权重。这就像每层只有一个神经元，而且要慢得多。这种配置几乎不可能收敛到一个好的解决方案。
```


2.Is it OK to initialize the bias terms to 0?

```
将偏置项初始化为零是完全正确的。有些人喜欢像权重一样初始化它们，这也没关系; 它没有太大的区别。
```


3.Name three advantages of the SELU activation function over ReLU.

```
经过SELU激活函数后使得样本分布自动归一化到0均值和单位方差(自归一化，保证训练过程中梯度不会爆炸或消失，效果比Batch Normalization 要好) 
relu激活函数在负半轴坡度平缓，这样在activation的方差过大的时候可以让它减小，防止了梯度爆炸，但是正半轴坡度简单的设成了1。
而selu的正半轴大于1，在方差过小的的时候可以让它增大，同时防止了梯度消失。这样激活函数就有一个不动点，网络深了以后每一层的输出都是均值为0方差为1。
```


4.In which cases would you want to use each of the following activation functions: SELU, leaky ReLU (and its variants), ReLU, tanh, logistic, and softmax?

```
1.SELU经过该激活函数后使得样本分布自动归一化到0均值和单位方差。

2.如果你需要尽可能快地使用神经网络，则可以使用其中一个变体 leaky ReLU
（例如，使用默认超参数值的简单 leaky ReLU）。

3.ReLU激活函数的简单性使其成为许多人的首选选项，尽管它们通常优于ELU和 leaky ReLU。但是，在某些情况下，ReLU激活功能输出精确为零的能力可能很有用。

4.如果你需要输出介于-1和1之间的数字，则双曲正切（tanh）在输出层中很有用，但现在它在隐藏层中使用的次数不多。

5.当你需要估计概率时，逻辑激活函数在输出层中也很有用（例如，对于二元分类），但它也很少用于隐藏层。

6.softmax激活函数在输出层中用于输出互斥类的概率，但除此之外，很少（如果曾经）在隐藏层中使用它。
```


5.What may happen if you set the momentum hyperparameter too close to 1 (e.g., 0.99999) when using an SGD optimizer?

```
算法可能会获得很大的速度，希望大致达到全局最小值，但由于它的动量，它将会在最小值之后超调。 然后它会减速然后回来，再次加速，再次超调，等等。

在收敛之前它可能会以这种方式振荡很多次，因此总体而言，收敛所需的时间要比使用较小的动量值要长得多。
```


6.Name three ways you can produce a sparse model.

```
1.一种方法是正常训练模型，然后将微小的权重归零。

2.可以在训练期间应用l1正则化，从而将优化器推向稀疏性。

3.使用TensorFlow的FTRLOptimizer类将l1正则化与双重平均dual averaging相结合。
```


7.Does dropout slow down training? Does it slow down inference (i.e., making predictions on new instances)? What about MC Dropout?

```
dropout会减慢训练的速度，一般来说，大约是两倍。然而，它对预测没有影响，因为它只在训练时打开。

蒙特卡洛样本的数量是一个可以调节的超参数。这个数越高，预测和不准确度的估计越高。但是，如果样本数翻倍，推断时间也要翻倍。
另外，样本数超过一定数量，提升就不大了。因此要取决于任务本身，在延迟和准确性上做取舍。
