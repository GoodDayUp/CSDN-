---
Tensorflow-keras实战（九）：Estimator实战
---

# 目录：

1.泰坦尼克问题和feature_column结合使用
2.预定义estimator使用
3.交叉验证特征实战

#### 1.泰坦尼克问题和feature_column结合使用
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

print(train_df.shape,eval_df.shape)


train_df.age.hist(bins = 20) 


train_df.sex.value_counts().plot(kind = 'barh')  
train_df.sex.value_counts().plot(kind = 'barv')
train_df['class'].value_counts().plot(kind = 'barh')


pd.concat([train_df,y_train],axis = 1).groupby('sex').survived.mean().plot(kind="barh")


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
    return dataset



train_dataset = make_dataset(train_df,y_train,batch_size=5)


for x,y in train_dataset.take(1):
    print(x,y)


#keras.layers.DenseFeature  把feature_columns和dataset结合
for x,y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_column = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_column)(x).numpy())



for x,y in train_dataset.take(1):
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())



model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(2,activation='softmax'),
])
model.compile(loss='sparse_categorical_crossentropy',
             optimizer = keras.optimizers.SGD(lr=0.01),
             metrics=['accuracy'])


# 1.model.fit
# 2.model->estimator->train

train_dataset = make_dataset(train_df,y_train,epochs=100)
eval_dataset = make_dataset(eval_df,y_eval,epochs=1,shuffle=False)
history = model.fit(train_dataset,
                    validation_data=eval_dataset,
                    steps_per_epoch=20,
                    validation_steps = 8,
                    epochs=100)




estimator = keras.estimator.model_to_estimator(model)
# input_fn   1.function
#2.return a.(features,labels)  b.dataset->(feature,label)
estimator.train(input_fn = lambda : make_dataset(
    train_df,y_train,epochs=100))   


#2.0bug   名字没有被保存下来
```

#### 2.预定义estimator使用
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
    return dataset



output_dir = 'baseline_model'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

baseline_estimator = tf.estimator.BaselineClassifier(
    model_dir = output_dir,n_classes=2)
baseline_estimator.train(input_fn=lambda : make_dataset(train_df,y_train,epochs=100))

baseline_estimator.evaluate(input_fn=lambda : make_dataset(
    eval_df,y_eval,epochs=1,shuffle=False,batch_size=20))




linear_output_dir = 'linear_model'
if not os.path.exists(output_dir):
    os.mkdir(linear_output_dir)
    
linear_estimator = tf.estimator.LinearClassifier(
    model_dir = linear_output_dir,n_classes=2,feature_columns = feature_columns)
linear_estimator.train(input_fn = lambda : make_dataset(train_df,y_train,epochs=100))


linear_estimator.evaluate(input_fn=lambda : make_dataset(
    eval_df,y_eval,epochs=1,shuffle=False))




dnn_output_dir = './dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
    
dnn_estimator = tf.estimator.DNNClassifier(
    model_dir = dnn_output_dir,n_classes=2,
    feature_columns = feature_columns,hidden_units = [128,128],
    activation_fn = tf.nn.relu,optimizer = 'Adam')
dnn_estimator.train(input_fn = lambda : make_dataset(train_df,y_train,epochs=100))


dnn_estimator.evaluate(input_fn=lambda : make_dataset(
    eval_df,y_eval,epochs=1,shuffle=False))

```



#### 3.交叉验证特征实战# 泰坦尼克问题
```
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
    
    
for numeric_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            numeric_column,dtype=tf.float32))
    
# cross feture:age:[1,2,3,4,5],gender:[male,female]
# age_x_gender:[(1,male),(2,male),...,(5,male),...,(5,female)]
#100000:100->hash(100000values)%100



feature_columns.append(
    tf.feature_column.indicator_column(
        tf.feature_column.crossed_column(
            ['age','sex'],hash_bucket_size=100)))


```







