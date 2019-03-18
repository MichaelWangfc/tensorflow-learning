在Tensorflow的官方文档中，Graph 被定义为“一些 Operation 和 Tensor 的集合”。
图（graph）是 tensorflow 用于表达计算任务的一个核心概念。
从前端（python）描述神经网络的结构，到后端在多机和分布式系统上部署，到底层 Device（CPU、GPU、TPU）上运行，都是基于图来完成。

tf中可以定义多个计算图，不同计算图上的张量和运算是相互独立的，不会共享。计算图可以用来隔离张量和计算，同时提供了管理张量和计算的机制。
计算图可以通过Graph.device函数来指定运行计算的设备，为TensorFlow充分利用GPU/CPU提供了机制。
一个图可以在多个sess中运行，一个sess也能运行多个图.with语句是保证操作的资源可以正确的打开和释放，而且不同的计算图上的张量和运算彼此分离，互不干扰.

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
tf.get_default_graph() #可以获取当前默认的计算图句柄。

g = tf.Graph() #创建新的计算图; 在Tensorflow中，始终存在一个默认的Graph，当你创建Operation、Tensor时，tensorflow会将你这些节点和边自动添加到这个默认Graph中。 
			   #那么，当你只想创建一个图时，并不需要使用tf.Graph()函数创建一个新图，而是直接定义需要的Tensor和Operation，
			   #这样，tensorflow会将这些节点和边自动添加到默认Graph中。

with g.as_default(): # 语句下定义属于计算图g的张量和操作,这样with语句块中调用的Operation或Tensor将会添加到该Graph中。
tf.Sesstion(graph= ) # 通过参数 graph = xxx指定当前会话所运行的计算图;

sess.graph

tf.Graph.device() #TensorFlow中计算图可以通过tf.Graph.device函数来指定运行计算图的设备。下面程序将加法计算放在CPU上执行，也可以使用tf.device达到同样的效果。
g = tf.Graph()
with g.device('/cpu:0'):
    result = 1 + 2

g.get_tensor_by_name() #在图里面可以通过名字得到对应的元素
g.get_operation_by_name()  #获取节点操作op的方法和获取张量的方法非常类似
g.get_operations() #返回图中的操作节点列表


#There are two ways to get the graph:
#1) Call the graph using tf.get_default_graph(), which returns the default graph of the program
#2) set it as sess.graph which returns the session’s graph (note that this requires us to have a session created).
import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))

	
	
##==========tensorflow中创建多个计算图(Graph)
# -*- coding: utf-8 -*-)
import tensorflow as tf
 
# 在系统默认计算图上创建张量和操作
a=tf.constant([1.0,2.0])
b=tf.constant([2.0,1.0])
result = a+b
 
# 定义两个计算图
g1=tf.Graph()
g2=tf.Graph()
 
# 在计算图g1中定义张量和操作
with g1.as_default():
    a = tf.constant([1.0, 1.0])
    b = tf.constant([1.0, 1.0])
    result1 = a + b
 
with g2.as_default():
    a = tf.constant([2.0, 2.0])
    b = tf.constant([2.0, 2.0])
    result2 = a + b
 
 
# 在g1计算图上创建会话
with tf.Session(graph=g1) as sess:
    out = sess.run(result1)
    print 'with graph g1, result: {0}'.format(out)
 
with tf.Session(graph=g2) as sess:
    out = sess.run(result2)
    print 'with graph g2, result: {0}'.format(out)
 
# 在默认计算图上创建会话
with tf.Session(graph=tf.get_default_graph()) as sess:
    out = sess.run(result)
    print 'with graph default, result: {0}'.format(out)
 
print g1.version  # 返回计算图中操作的个数

#with graph g1, result: [ 2.  2.]
#with graph g2, result: [ 4.  4.]
#with graph default, result: [ 3.  3.]
#3

###=============================Tensorflow 三种 API 所保存和恢复的图=============================
###==============================================================================================
###==============================================================================================
Tensorflow 三种 API 所保存和恢复的图是不一样的。这三种图是从Tensorflow框架设计的角度出发而定义的。
	tf.train.Saver()/saver.restore()
	tf.train.export_meta_graph/tf.train.import_meta_graph()
	tf.train.write_graph()/tf.Import_graph_def()

###============================tf.train.saver.save() =======================
方法：	tf.train.saver.save() 在保存check-point的同时也会保存Meta Graph。
		但是在恢复图时，tf.train.saver.restore() 只恢复 Variable，如果要从MetaGraph恢复图，需要使用 import_meta_graph
		这是其实为了方便用户，有时我们不需要从MetaGraph恢复的图，而是需要在 python 中构建神经网络图，并恢复对应的 Variable。
API：	tf.train.saver.save()
		tf.train.saver.restore()
		tf.train.import_meta_graph()

import tensorflow as tf
model = ...
saver = tf.train.saver()
with tf.Session() as sess:    
    saver.restore(sess,'./checkpoint_dir')
	#在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过已经保存的模型加载进来.

#如果要从MetaGraph恢复图，需要使用 import_meta_graph:
import tensorflow as tf
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
	#在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过已经保存的模型加载进来.

	
###============================tf.train.export_meta_graph========================================
方法：	通过 export_meta_graph 保存图，并且通过 add_to_collection 将 train_op 加入到 collection中：
API:	tf.train.export_meta_graph()
		tf.train.import_meta_graph 就是用来进行 Meta Graph 读写的API。

#通过 add_to_collection 将 train_op 加入到 collection中,通过export_meta_graph保存图
with tf.Session() as sess:
  pred = model_network(X)
  loss=tf.reduce_mean(…,pred, …)
  train_op=tf.train.AdamOptimizer(lr).minimize(loss)
  tf.add_to_collection("training_collection", train_op)
  Meta_graph_def = 
      tf.train.export_meta_graph(tf.get_default_graph(), 'my_graph.meta')
	  
#通过 import_meta_graph将图恢复（同时初始化为本 Session的 default 图），并且通过 get_collection 重新获得 train_op，以及通过 train_op 来开始一段训练（ sess.run() ）
with tf.Session() as new_sess:
  tf.train.import_meta_graph('my_graph.meta')
  train_op = tf.get_collection("training_collection")[0]
  new_sess.run(train_op)
  

###============================tf.train.write_graph()/tf.Import_graph_def()========================================



	
###=============================Tensorflow graph 重要概念========================================
###==============================================================================================
###==============================================================================================	
Graph_def &&  NodeDef 
MetaGraph
Check-point
	

###=============================Graph_def && NodeDef ==============================================
Graph_def：
	定义：	Tensorflow 中 Graph 和它的序列化表示 Graph_def。
			tf.Operation 的序列化 ProtoBuf 是 NodeDef，但grph_def 不包括tf.Variable，需要加载MetaGraph得到variable
			
	API:
		graph.as_graph_def()
		graph_pb2.GraphDef()
		
NodeDef:
	从 python Graph中序列化出来的图就叫做 GraphDef（这是一种不严格的说法，先这样进行理解）。
	而 GraphDef 又是由许多叫做 NodeDef 的 Protocol Buffer 组成。
	在概念上 NodeDef 与 （Python Graph 中的）Operation 相对应。如下就是 GraphDef 的 ProtoBuf，由许多node组成的图表示。
	这是与上文 Python 图对应的 GraphDef：	
	
	举例：	例如我们表达如下的一个计算的 python代码:
	a = tf.placeholder(tf.float32)
	b = tf.placeholder(tf.float32)
	c = tf.placeholder(tf.float32)
	d = a*b+c
	e = d*2

	就会生成相应的一张图，在Tensorboard中看到的图大概如下这样。其中每一个圆圈表示一个Operation（输入处为Placeholder），椭圆到椭圆的边为Tensor，箭头的指向表示了这张图
	Operation 输入输出 Tensor 的传递关系。

	这张图所表达的数据流 与 python 代码中所表达的计算是对应的关系（为了称呼方便，我们下面将这张由Python表达式所描述的数据流动关系叫做 Python Graph）。
	然而在真实的 Tensorflow 运行中，Python 构建的“图”并不是启动一个Session之后始终不变的东西。因为Tensorflow在运行时，真实的计算会被下放到多CPU上，
	或者 GPU 等异构设备，或者ARM等上进行高性能/能效的计算。单纯使用 Python 肯定是无法有效完成的。
	实际上，Tensorflow而是首先将 python 代码所描绘的图转换（即“序列化”）成 Protocol Buffer，再通过 C/C++/CUDA 运行 Protocol Buffer 所定义的图。
	Protocol Buffer的介绍可以参考这篇文章学习：https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/）

	
node {
  name: "Placeholder"     # 注释：这是一个叫做 "Placeholder" 的node
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "Placeholder_1"     # 注释：这是一个叫做 "Placeholder_1" 的node
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        unknown_rank: true
      }
    }
  }
}
node {
  name: "mul"                 # 注释：一个 Mul（乘法）操作
  op: "Mul"
  input: "Placeholder"        # 使用上面的node（即Placeholder和Placeholder_1）
  input: "Placeholder_1"      # 作为这个Node的输入
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}


那么既然 tf.Operation 的序列化 ProtoBuf 是 NodeDef，那么 tf.Variable 呢？
在这个 GraphDef 中只有网络的连接信息，却没有任何 Variables呀？
没错，Graphdef中不保存任何 Variable 的信息，所以如果我们从 graph_def 来构建图并恢复训练的话，是不能成功的。
比如以下代码:
#tf.train.write_graph()/tf.Import_graph_def() 就是用来进行 GraphDef 读写的API。
with tf.Graph().as_default() as graph:
  tf.import_graph_def("graph_def_path")
  saver= tf.train.Saver()
  with tf.Session() as sess:
    tf.trainable_variables()

其中 tf.trainable_variables() 只会返回一个空的list。Tf.train.Saver() 也会报告 no variables to save。
	
然而，在实际线上 inference 中，通常就是使用 GraphDef。然而，GraphDef中连Variable都没有，怎么存储weight呢？
原来GraphDef 虽然不能保存 Variable，但可以保存 Constant 。
通过 tf.constant 将 weight 直接存储在 NodeDef 里，tensorflow 1.3.0 版本也提供了一套叫做 freeze_graph 的工具来自动的将图中的 Variable 替换成 
constant 存储在 GraphDef 里面，并将该图导出为 Proto。可以查看以下链接获取更多信息，	

###=============================MetaGraph &&& MetaGraphDef =============================== 
MetaGraph：
	定义：	Meta graph 的官方解释是：一个Meta Graph 由一个计算图和其相关的元数据构成。其包含了用于继续训练，实施评估和（在已训练好的的图上）做前向推断的信息。
			（A MetaGraph consists of both a computational graph and its associated metadata. A MetaGraph contains the information required to 
			continue training, perform evaluation, or run inference on a previously trained graph. 
			From <https://www.tensorflow.org/versions/r1.1/programmers_guide/> ）
	
	作用：	从序列化的图中，得到 Variables

MetaGraphDef
	作用：	Meta Graph在具体实现上就是一个 MetaGraphDef （同样是由 Protocol Buffer来定义的）。
			Meta Graph中虽然包含Variable的信息，却没有 Variable 的实际值。所以从Meta Graph中恢复的图，其训练是从随机初始化的值开始的。
			训练中Variable的实际值都保存在check-point中，如果要从之前训练的状态继续恢复训练，就要从check-point中restore。
			
			Meta Graph包含了四种主要的信息，根据Tensorflow官网，这四种 Protobuf 分别是:
			MetaInfoDef，存一些元信息（比如版本和其他用户信息）
			GraphDef， MetaGraph的核心内容之一，我们刚刚介绍过
			SaverDef，图的Saver信息（比如最多同时保存的check-point数量，需保存的Tensor名字等，但并不保存Tensor中的实际内容）
			CollectionDef 任何需要特殊注意的 Python 对象，需要特殊的标注以方便import_meta_graph 后取回。（比如“train_op”,"prediction"等）
				Collection（CollectionDef是对应的ProtoBuf）。
				Tensorflow中并没有一个官方的定义说 collection 是什么。简单的理解，它就是为了方别用户对图中的操作和变量进行管理，而创建的一个概念。它可以说是一种“集合”，
				通过一个 key（string类型）来对一组 Python 对象进行命名的集合。这个key既可以是tensorflow在内部定义的一些key，也可以是用户自己定义的名字（string）。

				Tensorflow 内部定义了许多标准 Key，全部定义在了 tf.GraphKeys 这个类中。其中有一些常用的，tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES 等等。
				tf.trainable_variables() 与 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 是等价的；tf.global_variables() 与 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
				是等价的。
	
#对于用户定义的 key，例如：
pred = model_network(X)
loss=tf.reduce_mean(…, pred, …)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)
#这样一段 Tensorflow程序，用户希望特别关注 pred, loss train_op 这几个操作，那么就可以使用如下代码，将这几个变量加入到 collection 中去。
#(假设我们将其命名为 “training_collection”)

tf.add_to_collection("training_collection", pred)
tf.add_to_collection("training_collection", loss)
tf.add_to_collection("training_collection", train_op)

并且可以通过 Train_collect = tf.get_collection(“training_collection”) 得到一个python list，其中的内容就是 pred, loss, train_op的 Tensor。
这通常是为了在一个新的 session 中打开这张图时，方便我们获取想要的操作。比如我们可以直接通过get_collection() 得到 train_op，
然后通过 sess.run(train_op)来开启一段训练，而无需重新构建 loss 和optimizer。





