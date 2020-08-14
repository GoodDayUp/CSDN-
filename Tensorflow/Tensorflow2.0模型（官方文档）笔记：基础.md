---
Tensorflow2.0模型（官方文档）笔记：基础
---

在 TensorFlow 2.0 版本中，Eager Execution 模式为默认模式，无需额外调
```tf.enable_eager_execution() ```函数（不过若要关闭 Eager Execution，则需调用```tf.compat.v1.disable_eager_execution()```函数）。


TensorFlow 的大多数 API 函数会根据输入的值自动推断张量中元素的类型（一般默认为```tf.float32```）。不过你也可以通过加入```dtype```参数来自行指定类型，例如```zero_vector = tf.zeros(shape=(2), dtype=tf.int32)```将使得张量中的元素类型均为整数。张量的```numpy()```方法是将张量的值转换为一个 NumPy 数组。


TensorFlow 提供了强大的 自动求导机制 来计算导数。以下代码展示了如何使用```tf.GradientTape()```计算函数 y(x)=x^2 在 x=3 时的导数：
```
import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y, y_grad])
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191031101624640.png)

```
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])
```

这里，```tf.square()```操作代表对输入张量的每一个元素求平方，不改变张量形状。```tf.reduce_sum()```操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量（可以通过 axis 参数来指定求和的维度，不指定则默认对所有元素求和）。



以下展示了如何使用 TensorFlow 计算线性回归。可以注意到，程序的结构和前述 NumPy 的实现非常类似。这里，TensorFlow 帮助我们做了两件重要的工作：

1.使用```tape.gradient(ys, xs)```自动计算梯度；

2.使用```optimizer.apply_gradients(grads_and_vars)```自动更新模型参数。


```
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
```





