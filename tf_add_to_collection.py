
'''
Tensorflow中并没有一个官方的定义说 collection 是什么。简单的理解，它就是为了方别用户对图中的操作和变量进行管理，而创建的一个概念。
它可以说是一种“集合”，通过一个 key（string类型）来对一组 Python 对象进行命名的集合。这个key既可以是tensorflow在内部定义的一些key，
也可以是用户自己定义的名字（string）。

Tensorflow 内部定义了许多标准 Key，全部定义在了 tf.GraphKeys 这个类中。其中有一些常用的，
tf.GraphKeys.TRAINABLE_VARIABLES, 
tf.GraphKeys.GLOBAL_VARIABLES 等等。

tf.trainable_variables() 与 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 是等价的；
tf.global_variables() 与 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 是等价的。

tf.add_to_collection，tf.get_collection和tf.add_n的用法

tf.add_to_collection(name, value)  用来把一个value放入名称是‘name’的集合，组成一个列表;

tf.get_collection(key, scope=None) 用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表，列表的顺序是按照变量放入集合中的先后;   scope参数可选，表示的是名称空间（名称域），如果指定，就返回名称域中所有放入‘key’的变量的列表，不指定则返回所有变量。

tf.add_n(inputs, name=None), 把所有 ‘inputs’列表中的所有变量值相加，name可选，是操作的名称。

'''

## coding: utf-8 ##
import tensorflow as tf
 
v1 = tf.get_variable(name='v1', shape=[1], initializer=tf.constant_initializer(1))
tf.add_to_collection('output', v1)  # 把变量v1放入‘output’集合中
v2 = tf.get_variable(name='v2', shape=[1], initializer=tf.constant_initializer(2))
tf.add_to_collection('output', v2)
v3 = tf.get_variable(name='v3', shape=[1], initializer=tf.constant_initializer(3))
tf.add_to_collection('output',v3)
 
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print tf.get_collection('output')    # 获取'output'列表内容
    print sess.run(tf.add_n(tf.get_collection('output')))  # tf.add_n把列表中所有内容一次性相加
 
 
# print:
# [<tf.Variable 'v1:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'v2:0' shape=(1,) dtype=float32_ref>, <tf.Variable 'v3:0' shape=(1,) dtype=float32_ref>]
# [ 6.]



