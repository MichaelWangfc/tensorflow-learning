import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

ckpt_dir = r'D:\wangfeicheng\Tensorflow\docker-multiple\ResNet\resnet50-cifar-ckpt-20190218'
file = os.path.join(ckpt_dir, 'checkpoint')
with open(file) as ckpt:
    lines = ckpt.readlines()
    # for x in lines:
    #     print(x)
print(lines)

# 查看TensorFlow checkpoint文件中的变量名和对应值
latest_ckpt_dir = os.path.join(ckpt_dir, 'model.ckpt-107738')
# print(latest_ckpt_dir)
reader = pywrap_tensorflow.NewCheckpointReader(latest_ckpt_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
print('Lenth of variables: %d' % len(var_to_shape_map))


# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(reader.get_tensor(key))


def create_config_proto():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    return config


tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    # Loading the graph first,then it can restore() the variable for this graph
    graph_dir = os.path.join(ckpt_dir, 'model.ckpt-107738.meta')

    # saver = tf.train.import_meta_graph(graph_dir)
    '''
    import_meta_graph appends the network defined in .meta file to the current graph. So, this will create the 
    graph/network for you but we still need to load the value of the parameters that we had trained on this graph.
    
    InvalidArgumentError (see above for traceback): Restoring from checkpoint failed. This is most likely due to a 
    mismatch between the current graph and the graph from the checkpoint. Please ensure that you have not altered the 
    graph expected based on the checkpoint. Original error:
    Cannot assign a device for operation IteratorToStringHandle: Operation was explicitly assigned to 
    /job:worker/task:0/device:GPU:0 but available devices are [ /job:localhost/replica:0/task:0/device:CPU:0 ].
     Make sure the device specification refers to a valid device.
     [[node IteratorToStringHandle (defined at D:/wangfeicheng/Tensorflow/tensorflow-learning/tf_saver.py:39)  = 
     IteratorToStringHandle[_device="/job:worker/task:0/device:GPU:0"](OneShotIterator)]]
    '''


    # saver = tf.train.Saver()  # Gets all variables in `graph`.

with tf.Session(graph=graph, config=create_config_proto()) as sess:
    # latest_ckpt_dir = tf.train.latest_checkpoint(ckpt_dir)
    #     # print(latest_ckpt_dir)
    saver.restore(sess, latest_ckpt_dir)
    # conv2d_51= graph.get_operation_by_name('conv2d_51/kernel')
    # print(sess.run(conv2d_51))
