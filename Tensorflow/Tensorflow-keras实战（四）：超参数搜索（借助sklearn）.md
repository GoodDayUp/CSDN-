---
title: Tensorflow-keras实战（四）：超参数搜索(借助sklearn)
date: 2019-9-7 13:56:40
tags: [tensorflow,keras,深度学习]
categories: 深度学习-tensorflow
---


# 目录

超参数搜索
1.为什么要超参数搜索
2.搜索策略
	2.1 网格搜索
	2.2 随机搜索
	2.3 遗传算法搜索
	2.4 启发式搜索
3.实战
使用scikit实现超参数搜索
3.1 手动实现hp搜索
3.2 借助sklearn实现hp搜索
3.3 实战sklearn hp搜索



## 1.为什么要超参数搜索
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122122777.png)
## 2.搜索策略

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122208597.png)
#### 2.1 网格搜索

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122448158.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
#### 2.2 随机搜索

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122603333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
#### 2.3 遗传算法

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122746841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
#### 2.4 启发式搜索

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190907122850100.png)
## 3. 实战

#### 使用scikit实现超参数搜索


**3.1 手动实现hp搜索**

```
# 手动实现超参数搜索
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


# learning rate: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# w = w+grad*learning rate

# 自己实现：1.模型简化，只有一个参数，现实中更多参数，可能好多层for循环 
# 2.for循环会默认只有上一个训练完才会考虑下一个，没有并行化处理

learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
histories = []
for lr in learning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30,activation="relu",
                            input_shape=x_train.shape[1:]),
        keras.layers.Dense(1),
    ])
    optimizer = keras.optimizers.SGD(lr)
    
    model.compile(loss = "mean_squared_error",optimizer = optimizer)
    callbacks = [keras.callbacks.EarlyStopping(
        patience=5, min_delta=1e-2)]
    history = model.fit(x_train_scaled,y_train,
                        validation_data=(x_valid_scaled,y_valid),
                        epochs = 100,
                        callbacks = callbacks)
    histories.append(history)



def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
for lr, history in zip(learning_rates,histories):
    print('learning_rates: ', lr)
    plot_learning_curves(history)
```

**3.2 借助sklearn实现hp搜索**

```
# RandomizerdSearchCV
# 1.转化为sklearn的model 
# 2. 定义参数搜索 
# 3. 搜索参数

def build_model(hidden_layers=1,layer_size=30,learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size,activation="relu",
                                input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers -1):
        model.add(keras.layers.Dense(layer_size,activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss = "mse",optimizer = optimizer)
    return model
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(
    build_model)
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]
history = sklearn_model.fit(x_train_scaled,y_train,epochs=100,
                            validation_data = (x_valid_scaled,y_valid),
                            callbacks = callbacks)
```

**3.3 实战sklearn hp搜索**

# 实战sklearn hp搜索

```
from scipy.stats import reciprocal
# f(x) = 1/(x*log(b/a))  a<=x<=b

param_distribution = {
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate":reciprocal(1e-4,1e-2),
}


from sklearn.model_selection import RandomizedSearchCV

random_search_cv = RandomizedSearchCV(sklearn_model,
                                        param_distribution,
                                        n_iter = 10,
                                        n_jobs = 1)
random_search_cv.fit(x_train_scaled,y_train,epochs = 100,
                    validation_data = (x_valid_scaled,y_valid),
                    callbacks = callbacks)


print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)


model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled, y_test)
```



