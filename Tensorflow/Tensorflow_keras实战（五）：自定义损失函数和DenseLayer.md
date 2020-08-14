---
Tensorflow-keras实战（五）：自定义损失函数和DenseLayer
---


# 目录
1.自定义损失函数
2.自定义DenseLayer
2.1 不带参
2.2 带参


### 1.自定义损失函数

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




# 自定义损失函数
def customized_mse(y_true, y_pred):
    return tf.reduce_mean(tf.squre(y_pred - y_true))

model = keras.models.Sequential([
    keras.layers.Dense(30,activation = 'relu',
                        input_shape = x_train.shape[1:]),
    keras.layers.Dense(1),
])
model.summary()
model.compile(loss = customized_mse,optimizer = "sgd",
                metrics = ["mean_squared_error"])
callbacks = [keras.callbacks.EarlyStopping(
    patience=5,min_delta=1e-2)]



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

### 2.自定义DenseLayer

#### 2.1 不带参

```
# lambda 自定义layer

# tf.nn.softplus:log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))
print(customized_softplus([-01.,-5.,0.,5.,10.]))
```

#### 2.2 带参

```
# 自定义DenseLyer

class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self,units,activation = None,**kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer,self).__init__(**kwargs)
    
    def bulid(self,input_shape):
        
        """构建所需要的参数"""
        # x*w+b  input_shape:[None,a] w:[a,b] output_shape:[None,b]
        
        self.kernel = self.add_weight(name = 'kernel',
                                    shape = (input_shape[1],self.units),
                                    initializer = 'uniform',
                                    trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                shape = (self.units, ),
                                initializer = 'zeros',
                                trainable = True)
        super(CustomizedDenseLayer,self).build(input_shape)
    
        
    def call(self,x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)
        

model = keras.models.Sequential([
    CustomizedDenseLayer(30,activation = 'relu',
                        input_shape = x_train.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    # keras.layers.Dense(1,activation = 'softplus'),
    # keras.layers.Dense(1),keras.layers.Activation('softplus'),
])
model.summary()
model.compile(loss = "mean_squared_error",optimizer = "sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5,min_delta=1e-2)]
```

