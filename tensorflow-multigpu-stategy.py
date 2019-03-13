'''
结构
Cluster是Job的集合，Job是Task的集合。
即：一个Cluster可以切分多个Job，一个Job指一类特定的任务，每个Job包含多个Task，比如parameter server(ps)、worker，在大多数情况下,一个机器上只运行一个Task.

在分布式深度学习框架中,我们一般把Job划分为Parameter Server和Worker:
参数服务器（ parameter server, ps ）:Parameter Job是管理参数的存储、获取和更新工作.
计算服务器（ worker ）:              Worker Job是来运行ops.

如果参数的数量太大,一台机器处理不了,这就要需要多个Tasks.


TF分布式模式

1）使用分布式TensorFlow 训练深度学习模型一般有两种方式。
In-graph 模式:计算图内分布式（ in-graph replication 〉。
使用这种分布式训练方式时，所有的任务都会使用一个TensorFlow 计算图中的变量（也就是深度学习模型中的参数），而只是将计算部分发布到不同的计算服务器上。
优点：
配置简单， 其他多机多GPU的计算节点，只要起个join操作， 暴露一个网络接口，等在那里接受任务就好了。这些计算节点暴露出来的网络接口，使用起来就跟本机的一个GPU的使用一样， 只要在操作的时候指定tf.device("/job:worker/task:n")，
就可以向指定GPU一样，把操作指定到一个计算节点上计算，使用起来和多GPU的类似。
多GPU 样例程序将计算复制了多份， 每一份放到一个GPU 上进行运算。但不同的GPU 使用的参数都是在一个TensorFlow 计算图中的。因为参数都是存在同一个计算图中，所以同步更新参数比
较容易控制。在12.3 节中给出的代码也实现了参数的同步更新。
缺点：
然而因为计算图内分布式需要有一个中心节点来生成这个计算图井分配计算任务，所以当数据量太大时，训练数据的分发依然在一个节点上， 要把训练数据分发到不同的机器上， 严重影响并发训练速度。
在大数据训练的情况下，这个中心节点容易造成性能瓶颈。 不推荐使用这种模式。


2）Between-graph 模式：计算图之间分布式( between-graph replication ）。
优点：
Between-graph模式下，训练的参数保存在参数服务器， 数据不用分发， 数据分片的保存在各个计算节点， 各个计算节点自己算自己的， 算完了之后， 把要更新的参数告诉参数服务器，参数服务器更新参数。
这种模式的优点是不用训练数据的分发了， 尤其是在数据量在TB级的时候， 节省了大量的时间，所以大数据深度学习还是推荐使用Between-graph模式。

使用这种分布式方式时，在每一个计算服务器上都会创建一个独立的TensorFlow 计算图，但不同计算图中的相同参数需要以一种固定的方式放到同一个参数服务器上。
tf.train.replica_device_setter 
函数来帮助完成这一个过程，在12.4.2 节中将给出具体的样例。因为每个计算服务器的TensorFlow 计算图是独立的，所以这种方式的并行度要更高。

缺点：
但在计算图之间分布式下进行参数的同步更新比较困难。
为了解决这个问题TensorFlow 提供了 tf.train.SyncRepIicasOptirnizer 函数来帮助实现参数的同步更新。这让计算图之间分布式方式被更加广泛地使用。


同步更新和异步更新

in-graph模式和between-graph模式都支持同步和异步更新。

在同步更新的时候， 每次梯度更新，要等所有分发出去的数据计算完成后，返回回来结果之后，把梯度累加算了均值之后，再更新参数。 
各个用于并行计算的电脑，计算完各自的batch 后，求取梯度值，把梯度值统一送到ps服务机器中，由ps服务机器求取梯度平均值，更新ps服务器上的参数。

好处：
loss的下降比较稳定， 
坏处：
处理的速度取决于最慢的那个分片计算的时间。


在异步更新的时候， 所有的计算节点，各自算自己的， 更新参数也是自己更新自己计算的结果， 
优点：
计算速度快， 计算资源能得到充分利用
缺点：
loss的下降不稳定， 抖动大。收敛曲线震动比较厉害，因为当A机器计算完更新了ps中的参数，可能B机器还是在用上一次迭代的旧版参数值。

数据量小的情况下， 各个节点的计算能力比较均衡的情况下， 推荐使用同步模式；
数据量很大，各个机器的计算性能掺差不齐的情况下，推荐使用异步的方式。

'''

#对于分布式 TensorFlow，我们首先需要了解它的基本原理。
#以下代码展示了如何构建一个最简单 TensorFlow 集群，以帮助我们理解它的基本原理。

#=====================================创建一个本地集群=================================
import tensorflow as tf
c = tf.constant("Hello, Distributed TensorFlow!")
# 创建一个本地TensorFlow集群
server = tf.train.Server.create_local_server()
# 在集群上创建一个会话
sess = tf.Session(server.target)
print(sess.run(c)）
#在以上代码中，我们先通过 tf.train.Server.create_local_server 在本地创建一个只有一台机器的 TensorFlow 集群。
#然后在集群上生成一个会话，通过该对话，我们可以将创建的计算图运行在 TensorFlow 集群上。
#虽然这只是一个单机集群，但它基本上反映了 TensorFlow 集群的工作流程。

#上面简单的案例只是一个任务的集群，若一个 TensorFlow 集群有多个任务时，我们需要使用 tf.train.ClusterSpec 来指定每一个任务的机器。


#==============================创建两个任务机器的集群========================
c = tf.constant("Hello from server1!")
#生成－个有两个任务的集群，一个任务跑在本地2222 端口，另外一个跑在本地2223 端口。
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
#通过上面生成的集群配置生成Server ，并通过job name 和task index 指定当前所启动的任务。该任务是第一个任务，所以task index 为0 。
server = tf.train.Server(cluster, job_name="local", task_index=0)
#通过server . target 生成会话来使用TensorFlow 集群中的资源
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) 
print sess.run(c)
server.join()


import tensorflow as tf
c = tf.constant("Hello from server2!")
#和第一个程序一样的集群配置。集群中的每一个任务需要采用相同的配置。
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
#指定task index 为1 ，所以这个程序将在localhost : 2223 启动服务。
server = tf.train.Server(cluster, job_name="local", task_index=1)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True)) 
print sess.run(c)
server.join()


#1、定义集群
#coding=utf-8
#多台机器，每台机器有一个显卡、或者多个显卡，这种训练叫做分布式训练
import  tensorflow as tf
#现在假设我们有A、B、C、D四台机器，首先需要在各台机器上写一份代码，并跑起来，各机器上的代码内容大部分相同
# ，除了开始定义的时候，需要各自指定该台机器的task之外。以机器A为例子，A机器上的代码如下：
cluster=tf.train.ClusterSpec({
    "worker": [
        "A_IP:2222",#格式 IP地址：端口号，第一台机器A的IP地址 ,在代码中需要用这台机器计算的时候，就要定义：/job:worker/task:0
        "B_IP:1234"#第二台机器的IP地址 /job:worker/task:1
        "C_IP:2222"#第三台机器的IP地址 /job:worker/task:2
    ],
    "ps": [
        "D_IP:2222",#第四台机器的IP地址 对应到代码块：/job:ps/task:0
    ]})

#2、在各台机器上，定义server
#比如A机器上的代码server要定义如下：
server=tf.train.Server(cluster,job_name='worker',task_index=0)#找到‘worker’名字下的，task0，也就是机器A

#3、在代码中，指定device
with tf.device('/job:ps/task:0'):#参数定义在机器D上
	w=tf.get_variable('w',(2,2),tf.float32,initializer=tf.constant_initializer(2))
	b=tf.get_variable('b',(2,2),tf.float32,initializer=tf.constant_initializer(5))
 
with tf.device('/job:worker/task:0/cpu:0'):#在机器A cpu上运行
	addwb=w+b
with tf.device('/job:worker/task:1/cpu:0'):#在机器B cpu上运行
	mutwb=w*b
with tf.device('/job:worker/task:2/cpu:0'):#在机器C cpu上运行
	divwb=w/b

#在深度学习训练中，一般图的计算，对于每个worker task来说，都是相同的，所以我们会把所有图计算、变量定义等代码，都写到下面这个语句下：
with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:indexi',cluster=cluster)):





