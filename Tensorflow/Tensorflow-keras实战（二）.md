---
title: Tensorflow-keras实战（二）
date: 2019-9-5 15:10:40
tags: [tensorflow,keras,深度学习]
categories: 深度学习-tensorflow
---


## 实战

1.keras实现深度神经网络
2.keras更改激活函数
3.keras实现批归一化
4.keras实现dropout

**1.keras实现深度神经网络**

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


fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000],x_train_all[5000:]
y_valid, y_train = y_train_all[:5000],y_train_all[5000:]

print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



# x = (x-u)/ std
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [none,28,28]->[none,784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)



model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))

model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = 'sgd',
                metrics = ['accuracy'])




# TensorBoard, earlystopping, ModelCheckpoint
logdir = './dnn-callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,
                                "fashion_mnist_model.h5")
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True),
    keras.callbacks.EarlyStopping(patience = 5,min_delta = 1e-3),
]
history = model.fit(x_train_scaled,y_train,epochs=10,
                        validation_data=(x_valid_scaled,y_valid),
                    callbacks = callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,3)
    plt.show()
    
plot_learning_curves(history)

#1. 参数众多，训练不充分
#2. 梯度消失-> 链式法则-> 复合函数f(g(x))


model.evaluate(x_test_scaled,y_test)
#69
```

**2.keras批归一化（BatchNormalization）**

```
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation="relu"))
    model.add(keras.layers.BatchNormalization())
    """
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    """
    
model.add(keras.layers.Dense(10,activation="softmax"))


model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = 'sgd',
                metrics = ['accuracy'])

#81
```

**3.keras更改激活函数（selu）**

```
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation="selu"))   
model.add(keras.layers.Dense(10,activation="softmax"))


model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = 'sgd',
                metrics = ['accuracy'])

#85
```

**4.keras实现dropout**

```
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100,activation="selu"))
model.add(keras.layers.AlphaDropout(rate=0.5))
# AlphaDropout:1.均值方差不变 2.归一化性质也不变
# model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10,activation="softmax"))


model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = 'sgd',
                metrics = ['accuracy'])

# 86
```


