---
hands-on-ml-with-sklearn-and-tf-第13章课后题
---



1.Why would you want to use the Data API?

```
1.深度学习系统经常需要在大数据集上训练，而内存放不下大数据集；
TensorFlow通过Data API，只需要创建一个数据集对象，告诉它去哪里拿数据，以及如何做转换就行。

2.Data API还可以从现成的文件（比如CSV文件）、固定大小的二进制文件、使用TensorFlow的TFRecord格式的文件（支持大小可变的记录）读取数据。

3.Data API还支持从SQL数据库读取数据。
```


2.What are the benefits of splitting a large dataset into multiple files?

```
1.将大型数据集拆分为多个文件后，可以在使用改组缓冲区将其改组为更细的级别之前，先对其进行粗略的改组。
2.它还使处理不适合单台计算机的庞大数据集成为可能。处理数千个小文件而不是一个大文件也更简单；例如，将数据分成多个子集会更容易。
3.最后，如果将数据拆分为分布在多个服务器上的多个文件，则可以同时从不同的服务器下载多个文件，从而提高了带宽利用率。
```


3.During training, how can you tell that your input pipeline is the bottleneck? What can you do to fix it?

```
可以使用TensorBoard可视化分析数据：如果未充分利用GPU，则您的输入管道可能会成为瓶颈。

可以通过确保它并行读取和预处理多个线程中的数据并确保预取几个批次来修复它。
可以尝试将数据集保存到多个TFRecord文件中，并在必要时提前进行一些预处理，以便在训练过程中无需即时进行处理。
使用具有更多CPU和RAM的计算机，并确保GPU带宽足够大。
```


4.Can you save any binary data to a TFRecord file, or only serialized protocol buffers?

```
TFRecord文件由一系列任意二进制记录组成：您可以在每个记录中绝对存储所需的任何二进制数据。
实际上，大多数TFRecord文件都包含序列化协议缓冲区的序列。
这使得可以受益于协议缓冲区的优点，例如可以轻松地跨多种平台和语言读取它们，并且以后可以以向后兼容的方式更新它们的定义。
```


5.Why would you go through the hassle of converting all your data to the Example protobuf format? Why not use your own protobuf definition?

```
Example protobuf格式的优点是TensorFlow提供了一些操作来解析它（tf.io.parse * example（）函数），而无需定义自己的格式。
它足够灵活以表示大多数数据集中的实例。

如果它不能满足自己的用例，则可以定义自己的协议缓冲区，使用protoc进行编译（设置--descriptor_set_out和--include_imports参数以导出protobuf描述符），然后使用tf.io.decode_proto（ ）函数来解析序列化的protobuf。
这更加复杂，需要与模型一起部署描述符，但是可以做到。
```


6.When using TFRecords, when would you want to activate compression? Why not do it systematically?

```
当需要网络传输的时候，因为压缩会使文件更小，从而减少下载时间。
```


7.Data can be preprocessed directly when writing the data files, or within the tf.data pipeline, or in preprocessing layers within your model, or using TF Transform. Can you list a few pros and cons of each option?

```
1.如果在创建数据文件时对数据进行预处理，则训练脚本将运行得更快，因为它不必即时执行预处理。
在某些情况下，预处理后的数据也会比原始数据小得多，因此可以节省一些空间并加快下载速度。
实例化预处理的数据（例如检查或存档）也可能会有所帮助。

缺点:
首先，如果您需要为每个变体生成经过预处理的数据集，则尝试各种预处理逻辑并不容易。
其次，如果要执行数据扩充，则必须实现数据集的许多变体，这将占用大量磁盘空间并花费大量时间来生成。
最后，训练有素的模型将需要预处理的数据，因此您必须在应用程序中调用模型之前添加预处理代码。

2.如果使用tf.data管道对数据进行了预处理，则调整预处理逻辑并应用数据扩充会容易得多。
另外，tf.data可以轻松构建高效的预处理管道（例如，使用多线程和预取）。

缺点：
以这种方式预处理数据会减慢训练速度。
每个训练实例将在每个时期进行一次预处理，而不是在创建数据文件时对数据进行一次预处理。
经过训练的模型仍然需要预处理的数据，如果在模型中添加了预处理层，则只需为培训和推理编写一次预处理代码即可。

3.如果模型需要部署到许多不同的平台，则无需多次编写预处理代码。
另外，不会冒为模型使用错误的预处理逻辑的风险，因为它将成为模型的一部分。

缺点：
对数据进行预处理将减慢训练速度，并且每个训练实例将在每个时期进行一次预处理。
默认情况下，预处理操作将在GPU上针对当前批次运行（将无法从CPU上的并行预处理和预取中受益）。
但是，Keras预处理层能够从预处理层中取消预处理操作，并将其作为tf.data管道的一部分运行，因此可以在CPU上的多线程执行和预取。

4.使用TF Transform进行预处理可为您提供先前选项的许多好处：
预处理的数据被实现，每个实例仅被预处理一次（加快训练速度），并且预处理层会自动生成，因此您只需编写一次预处理代码。
```


8.Name a few common techniques you can use to encode categorical features. What about text?

```
1.使用独热矢量编码类型特征
2.使用嵌入编码类型特征
```





