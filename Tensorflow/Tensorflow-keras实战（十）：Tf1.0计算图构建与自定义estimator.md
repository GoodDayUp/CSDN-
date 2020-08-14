---
Tensorflow-keras实战（十）：Tf1.0计算图构建与自定义estimator
---


# 目录：
1.Tf1.0计算图构建
2.Tf1.0dataset使用
2.1 使用 make_one_shot_iterator
2.2 使用make_initializable_iterator
3.tf1.0自定义estimator
### 1.Tf1.0计算图构建
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


print(np.max(x_train),np.min(x_train))



# x = (x-u)/ std
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [none,28,28]->[none,784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)


print(np.max(x_train_scaled),np.min(x_train_scaled))


hidden_units = [100,100]
class_num = 10

x = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.int64,[None])

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation = tf.nn.relu)
logits = tf.layers.dense(input_for_next_layer,
                         class_num)
# last_hidden_output * w(logits) -> softmax -> prob
# 1. logit -> softmax ->prob
# 2. labels -> one-hot
# 3. calculate cross entropy
loss = tf.losses.sparse_softmax_cross_entropy(labels = y,logits = logits)

# get accuracy
prediction = tf.argmax(logits,1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


print(x)
print(logits)


# session

init = tf.global_variables_initializer()
batch_size = 20
epochs = 10
train_steps_per_epoch = x_train.shape[0]//batch_size
valid_steps =  x_valid.shape[0]//batch_size


def eval_with_sess(sess,x,y,accuracy,images,labels,batch_size):
    eval_steps = images.shape[0] // batch_size
    eval_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step * batch_size : (step+1) * batch_size]
        batch_label = labels[step * batch_size : (step+1) * batch_size]
        accuracy_val = sess.run(accuracy,
                               feed_dict = {
                                   x:batch_data,
                                   y:batch_label
                               })
        eval_accuracies.append(accuracy_val)
    return np.mean(eval_accuracies)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train_scaled[
                step * batch_size : (step+1) * batch_size]
            batch_label = y_trian[
                step * batch_size : (step+1) * batch_size]
            loss_val,accuracy_val,_ = sess.run(
                [loss,accuracy,train_op],
                feed_dict = {
                    x:batch_train,
                    y:batch_label
                })
            
            print('\r[Train] epoch : %d, step: %d, loss: %3.5f, accuracy: %2.2f' % (
                epoch,step,loss_val,accuracy_val),end = "" )
            
        valid_accuracy = eval_with_sess(sess,x,y,accuracy,
                                        x_valid_scaled,y_valid,
                                        batch_size)
        
        print("\t[Valid] acc: %2.2f" % (valid_accuracy))


```


### 2.Tf1.0dataset使用

##### 2.1 使用 make_one_shot_iterator

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


print(np.max(x_train),np.min(x_train))


# x = (x-u)/ std
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# x_train: [none,28,28]->[none,784]
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28*28)


y_train = np.asarray(y_train,dtype = np.int64)
y_valid = np.asarray(y_valid,dtype = np.int64)
y_test = np.asarray(y_test,dtype = np.int64)



print(np.max(x_train_scaled),np.min(x_train_scaled))


def make_dataset(images,labels,epochs,batch_size,shuffle = True):
    dataset = tf.data.Dataset.from_tensor_slices((images,labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset



batch_size = 20
epochs = 10
dataset = make_dataset(x_train_scaled,y_train,
                       epochs = epochs,
                       batch_size = batch_size)

dataset_iter = dataset.make_one_shot_iterator()
x,y = dataset_iter.get_next()
with tf.Session() as sess:
    x_val,y_val = sess.run([x,y])
    print(x_val.shape)
    print(y_val.shape)


hidden_units = [100,100]
class_num = 10

input_for_next_layer = x
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation = tf.nn.relu)
logits = tf.layers.dense(input_for_next_layer,
                         class_num)
# last_hidden_output * w(logits) -> softmax -> prob
# 1. logit -> softmax ->prob
# 2. labels -> one-hot
# 3. calculate cross entropy
loss = tf.losses.sparse_softmax_cross_entropy(labels = y,logits = logits)

# get accuracy
prediction = tf.argmax(logits,1)
correct_prediction = tf.equal(prediction, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


print(x)
print(logits)


# session

init = tf.global_variables_initializer()
train_steps_per_epoch = x_train.shape[0]//batch_size

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            loss_val,accuracy_val,_ = sess.run(
                [loss,accuracy,train_op])
            
            print('\r[Train] epoch : %d, step: %d, loss: %3.5f, accuracy: %2.2f' % (
                epoch,step,loss_val,accuracy_val),end = "" )
 
```


##### 2.2 使用make_initializable_iterator

```
batch_size = 20
epochs = 10

images_placeholder = tf.placeholder(tf.float32,[None,28 * 28])
labels_placeholder = tf.placeholder(tf.int64,[None,])
dataset = make_dataset(images_placeholder,labels_placeholder,
                       epochs = epochs,
                       batch_size = batch_size)


dataset_iter = dataset.make_initializable_iterator()
x,y = dataset_iter.get_next()
with tf.Session() as sess:
    ses.run(dataset_iter.initializer,
           feed_dict = {
               images_placeholder:x_train_scaled,
               labels_placeholder:y_train
           })
    x_val,y_val = sess.run([x,y])
    print(x_val.shape)
    print(y_val.shape)
    ses.run(dataset_iter.initializer,
           feed_dict = {
               images_placeholder:x_valid_scaled,
               labels_placeholder:y_valid
           })
    print(x_val.shape)
    print(y_val.shape)


```


### 3.tf1.0自定义estimator
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

# 泰坦尼克问题
# https://storage.googleapis.com/tf-datasets/titanic/train.csv
# https://storage.googleapis.com/tf-datasets/titanic/eval.csv
train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)

print(train_df.head())
print(eval_df.head())



y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')

print(train_df.head())
print(eval_df.head())
print(y_train.head())
print(y_eval.head())

train_df.describe()



categorical_columns = ['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numeric_columns = ['age','fare']

feature_columns=[]
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column,vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column,vocab)))
    
    
for categorical_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            categorical_column,dtype=tf.float32))





def make_dataset(data_df,label_df,epochs=10,shuffle=True,batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
        dataset=dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()





def model_fn(features,labels,mode,params):
    # model runtime state:Train,Eval,Predict
    input_for_next_layer = tf.feature_column.input_layer(
        features,params["feature_columns"])
    for n_unit in params["hidden_units"]:
        input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                              units = n_unit,
                                              activation = tf.nn.relu)
    logits = tf.layers.dense(input_for_next_layer,
                            params["n_classes"],
                            activation = None)
    predicted_classes = tf.argmax(logits,1)
    if mode ==tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "class_ids":predicted_classes[:,tf.newaxis],
            "probabilities":tf.nn.softmax(logits),
            "logits":logits
        }
        return tf.estimator.EstimatorSpec(mode,
                                         predictions = predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels,
                                                 logits = logits)
    accuracy = tf.metrics.accuracy(labels = labels,
                                  predictions = predicted_classes,
                                  name = "acc_op")
    metrics = {"accuracy":accuracy}
    if mode ==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss = loss,
                                          eval_metric_ops = metrics)
    optimizer = tf.train.minimize(
        loss,global_step = tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,loss = loss,
                                          train_op = train_op)

estimator = tf.estimator.Estimator(
    model_fn = model_fn,
    model_dir = output_dir,
    params = {
        "feature_columns":feature_columns,
        "hidden_units":[100,100],
        "n_classes":2
    }
)

estimator.train(input_fn = lambda : make_dataset(train_df,y_train,epochs = 100))



estimator.evaluate(lambda : make_dataset(eval_df,y_eval,epochs = 1))
```



### 4.Tf1.0和Tf2.0区别
##### 4.1静态图和动态图
tf1.0:Sess、feed_dict、placeholder被移除
tf1.0:make_one_shot(initializable)_iterator被移除
tf2.0:eager mode,@tf.function与AutoGraph
**例如：**
```
# Tensorflow 1.0
outputs = session.run(f(placeholder),feed_dict = {
    placeholder:input})
# Tensorflow 2.0
outputs = f(input)
```

**tf.function与AutoGraph**

性能好；
可以导入为SavedModel
例如：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921142557227.png)

##### 4.2 API变动

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921142720785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921142758508.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921142907915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
##### 4.3 如何升级

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921143217722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190921143335568.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1hCX3BsZWFzZQ==,size_16,color_FFFFFF,t_70)
