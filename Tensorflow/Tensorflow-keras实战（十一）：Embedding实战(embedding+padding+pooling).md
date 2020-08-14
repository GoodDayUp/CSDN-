---
title: Tensorflow-keras实战（十一）：Embedding实战(embedding+padding+pooling)
date: 2019-9-30 14:23:20
tags: [tensorflow,keras,深度学习]
categories: 深度学习-tensorflow
---


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



# 数据集载入
imdb = keras.datasets.imdb
vocab_size = 10000
index_from = 3
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(
    num_words = vocab_size,index_from = index_from)

print(train_data[0],train_labels[0])
print(train_data.shape,train_labels.shape)
print(len(train_data[0]),len(train_data[1]))
print(test_data.shape,test_labels.shape)



# 构建词表索引
word_index = imdb.get_word_index()
print(len(word_index))
print(word_index)



word_index = {k:(v+3) for k,v in word_index.items()}



word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict([(value,key) for key,value in word_index.items()])

def decode_review(text_ids):
    return ' '.join(
        [reverse_word_index.get(word_id, "<UNK>") for word_id in text_ids])

decode_review(train_data[0])



# 数据padding
max_length = 500

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, # list of list
    value = word_index['<PAD>'],
    padding = 'post',# post, pre
    maxlen = max_length)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, # list of list
    value = word_index['<PAD>'],
    padding = 'post',# post, pre
    maxlen = max_length)

print(train_data[0])



# 模型构建与训练
embedding_dim = 16
batch_size = 128

model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]
    # 2. [1,2,3,4..], max_length * embedding_dim
    # 3. batch_size * max_length * embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim,
                           input_length = max_length),
    # batch_size * max_length * embedding_dim -> batch_size * embedding_dim
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64,activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])

model.summary()
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
             metrics = ['accuracy'])



history = model.fit(train_data,train_labels,
                   epochs = 30,
                   batch_size = batch_size,
                   validation_split = 0.2)



def plot_learning_curves(history,label,epochs,min_value,max_value):
    data = {}
    data[label] = history.history[label]
    data['val_'+label] = history.history['val_'+label]
    pd.DataFrame(data).plot(figsize=(8,5))
    plt.grid(True)
    plt.axis([0,epochs,min_value,max_value])
    plt.show()
    
plot_learning_curves(history,'accuracy',30,0,1)
plot_learning_curves(history,'loss',30,0,1)


model.evaluate(test_data,test_labels,
              batch_size = batch_size)
```
