---
title: Tensorflow基础API
date: 2019-9-8 15:02:40
tags: [tensorflow]
categories: 深度学习-tensorflow
---

#### 1.导入

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
```

#### 2.@tf.constant

```
t = tf.constant([[1.,2.,3.],[4.,5.,6.]])
# index
print(t)
print(t[:,1:])
print(t[..., 1])
print(t+10)
print(tf.square(t))
print(t @ tf.transpose(t))

# numpy conversion
print(t.numpy())
print(np.square(t))
np_t = np.array([[1.,2.,3.],[4.,5.,6.]])
print(tf.constant(np_t))

# Scalars 0维
t = tf.constant(2.781)
print(t.numpy())
print(t.shape)

# string
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
print(tf.strings.length(t,unit="UTF8_CHAR"))
print(tf.strings.unicode_decode(t,"UTF-8"))

# string array
t = tf.constant(["cafe","coffee","咖啡"])
print(tf.strings.length(t,unit = "UTF8_CHAR"))
r = tf.strings.unicode_decode(t,"UTF-8")
print(r) 

# RaggedTensor是不完整的n维矩阵
# ragged tensor
r = tf.ragged.constant([[11,12],[21,22,32],[],[41]])
#op
print(r)
print(r[1])
print(r[1:2])

r2 = tf.ragged.constant([[51,52],[],[71]])
print(tf.concat([r,r2],axis = 0))

r3 = tf.ragged.constant([[13,14],[21,32,43],[],[33]])
print(tf.concat([r,r3],axis = 1))


#raged tensor->tensor
# 0在正向值后边
print(r.to_tensor())  

# sparse tensor  ：indices必须排好序,否则调用不了to_dense
# 0随意位置（稀疏矩阵）
s = tf.SparseTensor(indices = [[0,1],[1,0],[2,3]],
                    values = [1.,2.,3.],
                    dense_shape=[3,4])
print(s)
print(tf.sparse.to_dense(s))



s2 = s*2.0
print(s2)

try:
    s3 = s+1
except TypeError as ex:
    print(ex)
    
s4 = tf.constant([[10.,20,],
                    [30.,40],
                    [50.,60],
                    [70.,80]])
print(tf.sparse.sparse_dense_matmul(s,s4))

# sparse tensor
# 不排序
s5 = tf.SparseTensor(indices = [[0,2],[0,1],[2,3]],
                    values = [1.,2.,3.],
                    dense_shape=[3,4])
print(s5)
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))



# Variables
v = tf.Variable([[1.,2.,3.],[4.,5.,6.]])
print(v)
print(v.value())
print(v.numpy())


# assign value   可对变量重新赋值
v.assign(2*v)
print(v.numpy())
v[0,1].assign(42)
print(v.numpy())
v[1]..assign([7.,8.,9.])
print(v.numpy())

try:
    v[1]=[7.,8.,9.]
except TypeError as ex:
    print(ex)
```