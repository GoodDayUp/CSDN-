---
title: Tensorflow2.0模型（官方文档）笔记：模型类的建立
date: 2019-10-31 12:23:20
tags: [tensorflow]
categories: 深度学习-tensorflow
---

Keras 有两个重要的概念： 模型（Model） 和 层（Layer） 。层将各种计算流程和变量进行了封装（例如基本的全连接层，CNN 的卷积层、池化层等），而模型则将各种层进行组织和连接，并封装成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。在需要模型调用的时候，使用 ```y_pred = model(X) ```的形式即可。

Keras 模型以类的形式呈现，我们可以通过继承 tf.keras.Model 这个 Python 类来定义自己的模型。

```
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 此处添加初始化代码（包含call方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031105949593.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

简单的线性模型 ```y_pred = a * X + b ```，我们可以通过模型类的方式编写如下：

```
import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)
```


**Keras的全连接层：线性变换 + 激活函数**


全连接层 (Fully-connected Layer，```tf.keras.layers.Dense```) 是 Keras 中最基础和常用的层之一，对输入矩阵 A  进行 f(AW + b) 的线性变换 + 激活函数操作。如果不指定激活函数，即是纯粹的线性变换 AW + b 。具体而言，给定输入张量```input = [batch_size, input_dim]```，该层对输入张量首先进行```tf.matmul(input, kernel) + bias```的线性变换 （ ```kernel```和 ```bias ```是层中可训练的变量），然后对线性变换后张量的每个元素通过激活函数 ```activation ```，从而输出形状为```[batch_size, units] ```的二维张量。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031110656211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031110758756.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
[1] Keras 中的很多层都默认使用 ```tf.glorot_uniform_initializer ```初始化变量，关于该初始化器可参考 https://tensorflow.google.cn/api_docs/python/tf/glorot_uniform_initializer。


[2] ```tf.matmul(input, kernel) ```的结果是一个形状为 ```[batch_size, units] ```的二维矩阵，这个二维矩阵要如何与形状为 ```[units] ```的一维偏置向量 ```[bias] ```相加呢？事实上，这里是 TensorFlow 的 Broadcasting 机制在起作用，该加法运算相当于将二维矩阵的每一行加上了 ```[bias]```。Broadcasting 机制的具体介绍可见 https://tensorflow.google.cn/xla/broadcasting。
