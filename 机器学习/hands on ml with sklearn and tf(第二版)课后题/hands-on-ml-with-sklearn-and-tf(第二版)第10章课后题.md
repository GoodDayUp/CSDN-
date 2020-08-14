---
hands-on-ml-with-sklearn-and-tf-第10章课后题
---




1. Draw an ANN using the original artificial neurons (like the ones in Figure 10-3)
that computes A ⊕ B (where ⊕ represents the XOR operation). Hint: A ⊕ B = (A
∧ ¬ B) ∨ (¬ A ∧ B).

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200330192626945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)


3. Why is it generally preferable to use a Logistic Regression classifier rather than a
classical Perceptron (i.e., a single layer of threshold logic units trained using the
Perceptron training algorithm)? How can you tweak a Perceptron to make it
equivalent to a Logistic Regression classifier?

```
只有当数据集可线性分离时，经典感知器才会收敛，并且它无法估计类概率。

相反，即使数据集不是线性可分的，Logistic回归分类器也会收敛到一个好的解决方案，并且它可以输出类概率。

如果你将Perceptron的激活函数更改为逻辑激活函数（或者如果有多个神经元则更改softmax激活函数），
如果你使用Gradient Descent（或其他一些最小化成本函数的优化算法，通常是交叉熵）训练它，那么它就等同于Logistic回归分类器。

```


4. Why was the logistic activation function a key ingredient in training the first MLPs?

```
逻辑激活函数是训练第一个MLP的关键因素，因为它的导数总是非零，因此梯度下降总是可以向下滚动。 
当激活函数是阶梯函数时，梯度下降不能移动，因为根本没有斜率。
```


5. Name three popular activation functions. Can you draw them?

```
阶梯函数，逻辑函数，双曲正切，整流线性单元。
```


6. Suppose you have an MLP composed of one input layer with 10 passthrough neurons, 
followed by one hidden layer with 50 artificial neurons, and finally one
output layer with 3 artificial neurons. All artificial neurons use the ReLU activation function.

• What is the shape of the input matrix X?
```
形状是m×10，其中m表示训练批量大小。
```
• What about the shape of the hidden layer’s weight vector Wh, and the shape of its bias vector bh?
```
W_h的形状为10×50，b_h的长度为50。
```
• What is the shape of the output layer’s weight vector Wo, and its bias vector bo?
```
W_o的形状为50×3，b_o的长度为3。
```
• What is the shape of the network’s output matrix Y?
```
mx3
```
• Write the equation that computes the network’s output matrix Y as a function of X, Wh, bh, Wo and bo.
```
Y =（X·W_h + b_h）·W_o + b_o
```


7. How many neurons do you need in the output layer if you want to classify email into spam or ham? 

```
只需要在神经网络的输出层中有一个神经元。
```
What activation function should you use in the output layer?
```
在估计概率时，通常会在输出层中使用逻辑激活函数。
```
If instead you want to tackle MNIST, how many neurons do you need in the output layer, using what activation function? 
```
需要输出层中有10个神经元，你必须用softmax激活函数替换逻辑函数，该函数可以处理多个类，每个类输出一个概率。
```
Answer the same questions for getting your network to predict housing prices as in Chapter 2.
```
需要一个输出神经元，在输出层根本不使用激活功能。
```


8. What is backpropagation and how does it work? What is the difference between backpropagation and reverse-mode autodiff?

```
反向传播是一种用于训练人工神经网络的技术。
它首先根据每个模型参数（所有权重和偏差）计算成本函数的梯度，然后使用这些梯度执行梯度下降步骤。
该反向传播步骤通常使用许多训练批次执行数千或数百万次，直到模型参数收敛到最小化成本函数的值。
为了计算梯度，反向传播使用reverse-mode autodiff。
反向模式自动编排执行正向传递计算图，计算当前训练批的每个节点的值，然后执行反向传递，一次计算所有梯度）。

反向传播是指使用多个反向传播步骤训练人工神经网络的整个过程，每个反向传播步骤计算梯度并使用它们执行梯度下降步骤。
相比之下，reverse-mode autodiff只是一种有效计算梯度的技术，它恰好被反向传播使用。
```


9. Can you list all the hyperparameters you can tweak in an MLP? 

```
隐藏层的数量、每个隐藏层中的神经元数量、每个隐藏层和输出层中使用的激活函数。
```
If the MLP overfits the training data, how could you tweak these hyperparameters to try to solve the problem?
```
减少隐藏层的数量并减少每个隐藏层的神经元数量。
```


10. Train a deep MLP on the MNIST dataset and see if you can get over 98% precision. 
Try adding all the bells and whistles (i.e., save checkpoints, use early stopping, 
plot learning curves using TensorBoard, and so on).

```
略
```


