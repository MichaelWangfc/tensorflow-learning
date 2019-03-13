
'''
Tensorflow读取数据的一般方式有下面3种：
https://www.cnblogs.com/bicker/p/8424538.html


1. preloaded直接创建变量：在tensorflow定义图的过程中，创建常量或变量来存储数据

2. feed：在运行程序时，通过feed_dict传入数据
可以在tensorflow运算图的过程中，将数据传递到事先定义好的placeholder中。方法是在调用session.run函数时，通过feed_dict参数传入

3. reader从文件中读取数据：在tensorflow图开始时，通过一个输入管线从文件中读取数据
上面的两个方法在面对大量数据时，都存在性能问题。这时候就需要使用到第3种方法，文件读取，让tensorflow自己从文件中读取数据

	1)自己写原始代码
	2)slim数据读取接口
	2)tf.data.Dataset接口


步骤：
1.获取文件名列表list
2.创建文件名队列，调用tf.train.string_input_producer，参数包含：文件名列表，num_epochs【定义重复次数】，shuffle【定义是否打乱文件的顺序】
3.定义对应文件的阅读器>* tf.ReaderBase >* tf.TFRecordReader >* tf.TextLineReader >* tf.WholeFileReader >* tf.IdentityReader >* tf.FixedLengthRecordReader
4. 解析器 >* tf.decode_csv >* tf.decode_raw >* tf.image.decode_image >* …
5. 预处理，对原始数据进行处理，以适应network输入所需
6. 生成batch，调用tf.train.batch() 或者 tf.train.shuffle_batch()
7. prefetch【可选】使用预加载队列slim.prefetch_queue.prefetch_queue()
8. 启动填充队列的线程，调用tf.train.start_queue_runners

读取文件格式举例
tensorflow支持读取的文件格式包括：CSV文件，二进制文件，TFRecords文件，图像文件，文本文件等等。具体使用时，需要根据文件的不同格式，选择对应的文件格式阅读器，再将文件名队列传为参数，传入阅读器的read方法中。方法会返回key与对应的record value。将value交给解析器进行解析，转换成网络能进行处理的tensor。

CSV文件读取：
阅读器：tf.TextLineReader
解析器：tf.decode_csv

二进制文件读取：
阅读器：tf.FixedLengthRecordReader
解析器：tf.decode_raw

图像文件读取：
阅读器：tf.WholeFileReader
解析器：tf.image.decode_image, tf.image.decode_gif, tf.image.decode_jpeg, tf.image.decode_png

TFRecords文件读取
TFRecords文件是tensorflow的标准格式。要使用TFRecords文件读取，事先需要将数据转换成TFRecords文件，具体可察看：convert_to_records.py 在这个脚本中，先将数据填充到tf.train.Example协议内存块(protocol buffer)，将协议内存块序列化为字符串，再通过tf.python_io.TFRecordWriter写入到TFRecords文件中去。
阅读器：tf.TFRecordReader
解析器：tf.parse_single_example

又或者使用slim提供的简便方法：slim.dataset.Data以及slim.dataset_data_provider.DatasetDataProvider方法	
'''

# 导入tensorflow
import tensorflow as tf 

# 新建一个Session
with tf.Session() as sess:
    # 我们要读三幅图片A.jpg, B.jpg, C.jpg
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    # string_input_producer会产生一个文件名队列
    filename_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=5)
    # reader从文件名队列中读数据。对应的方法是reader.read
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
    tf.local_variables_initializer().run()
    # 使用start_queue_runners之后，才会开始填充队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image_data = sess.run(value)
        with open('read/test_%d.jpg' % i, 'wb') as f:
            f.write(image_data)
			
			
'''
slim数据读取接口
用slim读取数据分为以下几步：

1.给出数据来源的文件名并据此建立slim.Dataset，逻辑上Dataset中是含有所有数据的，当然物理上并非如此。
2.根据slim.Dataset建立一个DatasetDataProvider，这个class提供接口可以让你从Dataset中一条一条的去取数据
3. 通过DatasetDataProvider的get接口拿到获取数据的op，并对数据进行必要的预处理（如有）
4.利用从provider中get到的数据建立batch，此处可以对数据进行shuffle，确定batch_size等等
5.利用分好的batch建立一个prefetch_queue
6.prefetch_queue中有一个dequeue的op，没执行一次dequeue则返回一个batch的数据。


slim提供的数据读取接口其实也不够简洁，看看生一部分的六个步骤就知道过程还有有些繁琐的，想要熟练运用，不了解一些Tensorflow的实现是有点难的。但是tf.data.Dataset则不然，他隐藏了所有Tensorflow处理数据流的细节，用户只需要几步简单的操作就可以轻松读到数据，这使得数据读取更加容易上手且写出的代码更加简洁、易懂。

'''

