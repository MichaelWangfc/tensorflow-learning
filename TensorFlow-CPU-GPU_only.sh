
'''
Keras如果是使用Theano后端的话，应该是自动不使用ＧＰＵ只是用ＣＰＵ的，启动ＧＰＵ使用Theano内部命令即可。 
对于Tensorflow后端的Keras以及Tensorflow会自动使用可见的ＧＰＵ，而我需要其必须只运行在ＣＰＵ上。
在一套标准的系统上通常有多个计算设备. TensorFlow 支持 CPU 和 GPU 这两种设备. 我们用指定字符串 strings 来标识这些设备. 比如:
"/cpu:0": 机器中的 CPU
"/gpu:0": 机器中的 GPU, 如果你有一个的话.
"/gpu:1": 机器中的第二个 GPU, 以此类推...

在默认情况下，即使机器有多个CPU, Tensor Flow 也不会区分它们，所有的CPU 都使用/cpu:O 作为名称
只使用CPU:
原文：https://blog.csdn.net/silent56_th/article/details/72628606 
网上查到三种方法，最后一种方法对我有用，但也对三种都做如下记录：
'''

'''
1. 系统来为 operation 指派设备,使用tensorflow声明Session时的参数：
如果一个 TensorFlow 的 operation 中兼有 CPU 和 GPU 的实现, 当这个算子被指派设备时, GPU 有优先权. 比如matmul中 CPU 和 GPU kernel 函数都存在. 
那么在 cpu:0 和 gpu:0 中, matmul operation 会被指派给 gpu:0 .

为了获取你的 operations 和 Tensor 被指派到哪个设备上运行, 用 log_device_placement 新建一个 session, 并设置为 True.

关于tensorflow中Session中的部分参数设置，以及Keras如何设置其调用的Tensorflow的Session，可以参见Keras设定GPU使用内存大小(Tensorflow backend)。 
对于Tensorflow，声明Session的时候加入device_count={'gpu':0}即可，代码如下：
log_device_placement : 记录设备指派情况,程序会将运行每一个操作的设备输出到屏幕
allow_soft_placement :为了避免出现你指定的设备不存在这种情况, 你可以在创建的 session 里把参数 allow_soft_placement 设置为 True, 这样 tensorFlow 会自动选择一个存在并且支持的设备来运行 operation.
'''
import tensorflow as tf 
sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))

#对于Keras,则调用后端函数，设置其使用如上定义的Session即可，代码如下：
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

'''
2. 手工指派设备,使用tensorflow的 with tf.device('/cpu:0')函数。
如果你不想使用系统来为 operation 指派设备, 而是手工指派设备, 你可以用 with tf.device 创建一个设备环境, 这个环境下的 operation 都统一运行在环境指定的设备上.
简单操作就是把所有命令都放在前面所述的域里面。
'''
with tf.device('/cpu:0'):

#对于多线程以及ＧＰＵ内存设置等可以参见Keras设定GPU使用内存大小(Tensorflow backend)；更多详细内容请见tensorflow官网。

'''
#3 . 第三种是使用CUDA_VISIBLE_DEVICES命令行参数：
TensorFlow 默认会占用设备上的所有GP U 以及每个GPU 的所有显存。如果在一个TensorFlow 程序中只需要使用部分G PU ，可以通过设置CUDA VISIBLE DEVICES 环境变量来控制。
以下样例介绍了如何在运行时设置这个环境变量。
'''
#只使用第二块GPU (GPU 编号从0 开始）。在democode.py 中， 机器上的第二块GPU的名称变成/gpu : 0，在运行时所有/gpu:0的运算将被放在第二块GPU 上。
CUDA VISIBLE DEVICES=l python democode . py
#只使用第一块和第二块GPU.
CUDA_VISIBLE_DEVICES=0,l python demo_code.py

#补充
#4. TensorFlow 也支持在程序中设置环境变量， 以下代码展示了如何在程序中设置这些环境变量。
import os
#只使用第三块GPU.
os.environ ["CUDA VISIBLE DEVICES"]="2"
#不使用GPU
os.environ ["CUDA VISIBLE DEVICES"]="-1"



##TensorFlow 动态分配显存的方法 。
config = tf.ConfigProto ()
#让 TensorFlow 按需分配显存。
config.gpu_options.allow_growth = True
#或者直接按固定的比例分配。以下代码会占用所有可使用 GPU 的 40%显存。
# config.gpu_options.per process_gpu_memory_fraction = 0.4
session= tf.Session(config=config, ... )

'''
By default, TensorFlow maps nearly all of the GPU memory of all GPUs (subject to CUDA_VISIBLE_DEVICES) visible to the process. This is done to more efficiently use the relatively precious GPU memory resources on the devices by reducing memory fragmentation.
TensorFlow provides two Config options on the Session to control this.
The first is the allow_growth option, which attempts to allocate only as much GPU memory based on runtime allocations:
'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#The second method is the per_process_gpu_memory_fraction option, which determines the fraction of the overall amount of memory that each visible GPU should be allocated. For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
