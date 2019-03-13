####Tensorflow: how to save/restore a model?
'''
Saver的背景介绍
    我们经常在训练完一个模型之后希望保存训练的结果，这些结果指的是模型的参数，以便下次迭代的训练或者用作测试。Tensorflow针对这一需求提供了Saver类。
Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值 。
只要提供一个计数器，当计数器触发时，Saver类可以自动的生成checkpoint文件。这让我们可以在训练过程中保存多个中间结果。例如，我们可以保存每一步训练的结果。
为了避免填满整个磁盘，Saver可以自动的管理Checkpoints文件。例如，我们可以指定保存最近的N个Checkpoints文件。

'''

#saver=tf.train.Saver()
#saver.save()
#saver.restore()
'''
|--checkpoint_dir
|    |--checkpoint
|    |--model.ckpt-107738.meta
|    |--model.ckpt-107738.data-00000-of-00001
|    |--model.ckpt-107738.index

模型文件列表checkpoint
	举例：	checkpoint文件，
	格式：	该文件是个文本文件
	作用：	里面记录了保存的最新的checkpoint文件以及其它checkpoint文件列表。
			这个文件是tf.train.Saver类自动生成且自动维护的。在 checkpoint文件中维护了由一个tf.train.Saver类持久化的所有TensorFlow模型文件的文件名。
			当某个保存的TensorFlow模型文件被删除时，这个模型所对应的文件名也会从checkpoint文件中删除。
			checkpoint中内容的格式为CheckpointState Protocol Buffer.
			在inference时，可以通过修改这个文件，指定使用哪个model

计算图结构model.ckpt.meta
	举例:	model.ckpt-107738.meta
	作用:	保存的是图结构，meta文件是pb（protocol buffer）格式文件，包含变量、op、集合等。可以理解为神经网络的网络结构，
			TensorFlow通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。TensorFlow中元图是由MetaGraphDef Protocol Buffer定义的。
			MetaGraphDef 中的内容构成了TensorFlow持久化时的第一个文件。保存MetaGraphDef 信息的文件默认以.meta为后缀名，文件model.ckpt.meta中存储的就是元图数据。
			该文件可以被 tf.train.import_meta_graph 加载到当前默认的图来使用。

ckpt文件
	举例
	MyModel.data-00000-of-00001
	MyModel.index
	二进制文件，保存了所有的weights、biases、gradients等变量。在tensorflow 0.11之前，保存在.ckpt文件中。0.11后，通过两个文件保存,如：
	格式：  通过SSTable格式存储的，可以大致理解为就是一个（key，value）列表。
	作用：  model.ckpt文件保存了TensorFlow程序中每一个变量的取值，
			model.ckpt文件中列表的第一行描述了文件的元信息，比如在这个文件中存储的变量列表。列表剩下的每一行保存了一个变量的片段，变量片段的信息是通过SavedSlice Protocol Buffer定义的。
			SavedSlice类型中保存了变量的名称、当前片段的信息以及变量取值。
	API：	TensorFlow提供了tf.train.NewCheckpointReader类来查看model.ckpt文件中保存的变量信息
'''

###=====================保存Tensorflow模型============
#tensorflow 提供了tf.train.Saver类来保存模型，值得注意的是，在tensorflow中，变量是存在于Session环境中，也就是说，只有在Session环境下才会存有变量值，
#因此，保存模型时需要传入session：
saver = tf.train.Saver()
saver.save(sess,"./checkpoint_dir/MyModel")

#如果想要在1000次迭代后，再保存模型，只需设置global_step参数即可：
saver.save(sess, './checkpoint_dir/MyModel',global_step=1000)

#在实际训练中，我们可能会在每1000次迭代中保存一次模型数据，但是由于图是不变的，没必要每次都去保存，可以通过如下方式指定不保存图：
saver.save(sess, './checkpoint_dir/MyModel',global_step=step,write_meta_graph=False)

#如果你希望每2小时保存一次模型，并且只保存最近的5个模型文件：
tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)


#如果我们不对tf.train.Saver指定任何参数，默认会保存所有变量。如果你不想保存所有变量，而只保存一部分变量，可以通过指定variables/collections。
#在创建tf.train.Saver实例时，通过将需要保存的变量构造list或者dictionary，传入到Saver中：
import tensorflow as tf
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1,w2])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, './checkpoint_dir/MyModel',global_step=1000)


#####=================重构恢复（构造）网络图
# tensorflow将图和变量数据分开保存为不同的文件。因此，在导入模型时，也要分为2步：构造网络图和加载参数
####1.恢复（构造）网络图
#一个比较笨的方法是，手敲代码，实现跟模型一模一样的图结构。其实，我们既然已经保存了图，那就没必要在去手写一次图结构代码。
saver=tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')

#2 加载参数
#仅仅有图并没有用，更重要的是，我们需要前面训练好的模型参数（即weights、biases等），本文第2节提到过，变量值需要依赖于Session，因此在加载参数时，先要构造好Session：

import tensorflow as tf
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoint_dir'))

#此时，W1和W2加载进了图，并且可以被访问：
import tensorflow as tf
with tf.Session() as sess:    
    saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./checkpoint_dir'))
	#在加载模型的代码中没有运行变量的初始化过程，而是将变量的值通过已经保存的模型加载进来.
    print(sess.run('w1:0'))
##Model has been restored. Above statement will print the saved value

###################### tf.train.get_checkpoint_state() 和 tf.train.latest_checkpoint()
tf.train.get_checkpoint_state 函数通过checkpoint文件找到模型文件名。
ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir,latest_filename=None)
该函数返回的是checkpoint文件CheckpointState proto类型的内容，其中有model_checkpoint_path和all_model_checkpoint_paths两个属性。
model_checkpoint_path保存了最新的tensorflow模型文件的文件名，
all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的文件名。

saver.restore(sess, ckpt_state.model_checkpoint_path)
# Assuming model_checkpoint_path looks something like:
#   /my-favorite-path/cifar10_train/model.ckpt-0,

#just restoring the ckpt data
saver.restore(sess,tf.train.latest_checkpoint(train_dir))
#以上两种读取checkpoint的方法，似乎只会读取训练保存时候的路径，因此当ckpt数据移动或者更改名称时候，造成无法读取

#读取拷贝后或者更改路径后的ckpt文件，
#train_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218'
saver.restore(sess, FLAGS.train_dir)

###=======save the check_point in distributed tensorflow
is_chief = (FLAGS.task_index == 0)
with tf.train.MonitoredTrainingSession(
	master=server.target,
	is_chief=is_chief,
	# change the checkpint_dir to FLAGS.train_dir
	# checkpoint_dir=FLAGS.log_root,
	checkpoint_dir=FLAGS.train_dir,

	# change save_checkpoint_secs into save_checkpoint_steps (frequency)
	# save_checkpoint_secs=60,
	save_checkpoint_steps=1000,

	#Add stop_hook into the hooks
	#hooks=[logging_hook, _LearningRateSetterHook()],
	hooks=[logging_hook, _LearningRateSetterHook(),stop_hook],

	#change the chief_only_hooks
	chief_only_hooks=[model.replicas_hook,summary_hook],
	#chief_only_hooks=[summary_hook],

	# Since we provide a SummarySaverHook, we need to disable default
	# SummarySaverHook. To do that we set save_summaries_steps to 0.
	save_summaries_steps=0,
	config=create_config_proto()) as mon_sess:
  while not mon_sess.should_stop():
	mon_sess.run(model.train_op)
	
	
###=========================restoreing
train_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218\\'
# train_dir="D:/wangfeicheng/Tensorflow/docker-multiple/ResNet/resnet50-cifar-ckpt-20190218/"


with tf.Session(config=create_config_proto(),graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    tf.logging.info('Building the graph')
    # images, labels = cifar_input.build_input(dataset, eval_data_dir, batch_size, mode)
    # model = ResNet(hps, images, labels, mode)

    X = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='Y')
    model = ResNet(hps, X, Y, mode)
    model.build_graph(istrain=False)

    try:
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
    except tf.errors.OutOfRangeError as e:
        tf.logging.error('Cannot restore checkpoint: %s', e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        tf.logging.info('No model to eval yet at %s', train_dir)

    tf.logging.info('Loading the checkpoint from train_dir:{0}'.format(ckpt_state.model_checkpoint_path))

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)