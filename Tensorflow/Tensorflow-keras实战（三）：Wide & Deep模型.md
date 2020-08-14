---
title: Tensorflow-keras实战（三）：Wide & Deep模型
date: 2019-9-5 17:08:40
tags: [tensorflow,keras,深度学习]
categories: 深度学习-tensorflow
---



# 目录
#### 理论
1.相关论文
2.稀疏特征及优缺点
3.密集特征及优缺点
4.Wide & deep vs. Wide及Wide & deep vs. Deep的模型结构
5.Google Play应用
#### 实战
1.功能API（函数式API）
2.子类API
3.多输入
4.多输出

# 理论

1.**Wide & Deep模型**

相关论文：https://arxiv.org/pdf/1606.07792v1.pdf

2.**稀疏特征**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190905151748908.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190905151859910.png)**稀疏特征的优缺点**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190905152133528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)3.**密集特征**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019090515251034.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)**密集特征的优缺点**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019090515255813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)4.**Wide & deep vs. Wide**

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019090515290567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)**Wide & deep vs. Deep**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190905152946625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)5.**Google Play应用**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190905153015668.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)

# 实战

Wide & Deep模型
1.功能API（函数式API）
2.子类API
3.多输入
4.多输出


**1.功能API**

```
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import os
import sklearn
import sys
import time
import tensorflow as tf

from tensorflow import keras
print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__, module.__version__)


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

from sklearn.model_selection import train_test_split

x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state = 7)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11)

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.fit_transform(x_valid)
x_test_scaled = scaler.fit_transform(x_test)


# 函数式API 功能API
input = keras.layers.Input(shape=x_train.shape[1:])
hidden1 = keras.layers.Dense(30,activation="relu")(input)
hidden2 = keras.layers.Dense(30,activation="relu")(hidden1)
#复合函数：f(x)=h(g(x))
concat = keras.layers.concatenate([input,hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs = [input],
                            outputs = [output])

model.summary()
model.compile(loss = "mean_squared_error",optimizer = 'sgd')
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]


history = model.fit(x_train_scaled,y_train,
                        validation_data=(x_valid_scaled,y_valid),
                    epochs = 100,
                    callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    
plot_learning_curves(history)


model.evaluate(x_test_scaled, y_test)
```

**2.子类API**

```
# 子类API

class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        """定义模型的层次"""
        self.hidden1_layer = keras.layers.Dense(30,activation="relu")
        self.hidden2_layer = keras.layers.Dense(30,activation="relu")
        self.output_layer = keras.layers.Dense(1)
        
    def call(self, input):
        """完成模型的正向计算"""
        hidden1 = self.hidden1_layer(input)
        hidden2 = self.hidden2_layer(hidden1)
        concat = keras.layers.concatenate([input,hidden2])
        output = self.output_layer(concat)
        return output

#model = WideDeepModel()
model = keras.models.Sequential([
    WideDeepModel(),
])

model.build(input_shape=(None, 8)) #(样本的数目，输入的fetch的数目)
model.summary()
model.compile(loss = "mean_squared_error",optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5,min_delta=1e-2)]
```

**3.多输入**

```
# 多输入
input_wide = keras.layers.Input(shape = [5])
input_deep = keras.layers.Input(shape = [6])
hidden1 = keras.layers.Dense(30,activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30,activation="relu")(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input_wide, input_deep],
                            outputs = [output])

model.summary()
model.compile(loss = "mean_squared_error",optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5,min_delta=1e-2)]
```

**4.多输出**

```
# 多输出
input_wide = keras.layers.Input(shape = [5])
input_deep = keras.layers.Input(shape = [6])
hidden1 = keras.layers.Dense(30,activation="relu")(input_deep)
hidden2 = keras.layers.Dense(30,activation="relu")(hidden1)
concat = keras.layers.concatenate([input_wide, hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs = [input_wide, input_deep],
                            outputs = [output,output2])

model.summary()
model.compile(loss = "mean_squared_error",optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5,min_delta=1e-2)]
```



