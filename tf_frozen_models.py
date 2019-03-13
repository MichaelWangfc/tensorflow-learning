####===================Frozen models 
####===================
####===================
'''
作用：	使用freeze_graph.py将模型文件和权重数据整合在一起并去除无关的Op
		tensorflow的Freezing，字面意思是冷冻，可理解为整合合并；就是将模型文件和权重文件整合合并为一个文件，主要用途是便于发布。
		“Protocol buffers are Google’s language-neutral, platform neutral, extensible mechanism for serializing structured data – think XML, but smaller, faster, and simpler.”
		The protobuf format allows TensorFlow models to be deployed on devices which do not have Python and TensorFlow installed
		Frozen models are used to combine graph definitions specified in graph.pbtxt files with the variables saved in checkpoints
		
（1）freeze_graph.py是怎么做的呢？
		 """Converts all variables in a graph and checkpoint into constants.
        （a）它先加载模型文件，
        （b）从checkpoint文件读取权重数据初始化到模型里的权重变量
        （c）将权重变量转换成权重常量 （因为常量能随模型一起保存在同一个文件里），
        （d）再通过指定的输出节点将没用于输出推理的Op节点从图中剥离掉，再重新保存到指定的文件里（用write_graphdef或Saver）
（2）路径
		文件目录：tensorflow/python/tools/free_graph.py
		测试文件：tensorflow/python/tools/free_graph_test.py 这个测试文件很有学习价值

（3）参数
    总共有11个参数，一个个介绍下(必选： 表示必须有值；可选： 表示可以为空)：
    1、input_graph：A `GraphDef` file to load.
					（必选）模型文件，可以是二进制的pb文件，或文本的meta文件，用input_binary来指定区分（见下面说明）
    2、input_saver：A TensorFlow Saver file.
					（可选）Saver解析器。保存模型和权限时，Saver也可以自身序列化保存，以便在加载时应用合适的版本。主要用于版本不兼容时使用。可以为空，为空时用当前版本的Saver。
    3、input_binary：A Bool. True means input_graph is .pb, False indicates .pbtxt.
					（可选）配合input_graph用，为true时，input_graph为二进制.pb，为false时，input_graph为.pbtxt文件。默认False

    4、input_checkpoint：The prefix of a V1 or V2 checkpoint, with V2 taking
						  priority.  Typically the result of `Saver.save()` or that of
						  `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
						  V1/V2.
						（必选）检查点数据文件。训练时，给Saver用于保存权重、偏置等变量值。这时用于模型恢复变量值。

    5、output_node_names：The name(s) of the output nodes, comma separated.
						（必选）输出节点的名字，有多个时用逗号分开。用于指定输出节点，将没有在输出线上的其它节点剔除。

    6、restore_op_name：（可选）从模型恢复节点的名字。升级版中已弃用。默认：save/restore_all

    7、filename_tensor_name：（可选）已弃用。默认：save/Const:0

    8、output_graph：String where to write the frozen `GraphDef`.
					（必选）用来保存整合后的模型输出文件。

    9、clear_devices：A Bool whether to remove device specifications.
					（可选），默认True。指定是否清除训练时节点指定的运算设备（如cpu、gpu、tpu。cpu是默认）

    10、initializer_nodes：（可选）默认空。权限加载后，可通过此参数来指定需要初始化的节点，用逗号分隔多个节点名字。

    11、variable_names_blacklist：The set of variable names to omit converting to constants (optional).
							  （可先）默认空。变量黑名单，用于指定不用恢复值的变量，用逗号分隔多个变量名字。



  Args:
    variable_names_whitelist: The set of variable names to convert (optional, by
                              default, all variables are converted),
    input_meta_graph: A `MetaGraphDef` file to load (optional).
    input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
                           variables (optional).
    saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                      load, in string format.
    checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                        or saver_pb2.SaverDef.V2).
  Returns:
    String that is the location of frozen GraphDef.
  """
（4）用法
'''

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# Freeze model from checkpoint file
def freeze_from_checkpoint():
	checkpoint_dir = "./Checkpoints/"
	path = tf.train.latest_checkpoint("./Checkpoints/")
	input_graph_path = "./Checkpoints/graph.pbtxt"
	output_nodes = "prediction"
	restore_op = "save/restore_all"
	filename_tensor = "save/Const:0"
	output_name = "./Checkpoints/frozen_model.pb"
	freeze_graph.freeze_graph(input_graph_path, "",
			False, path, output_nodes,
			restore_op, filename_tensor,
			output_name, True, "")
			
# Optimize frozen.pb file for inference			
from tensorflow.python.tools import optimize_for_inference_lib
def optimize_frozen_file():
	frozen_graph_filename = "./Checkpoints/frozen_model.pb"
	with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	input_node_list = ["x"]
	output_node_list = ["prediction"]
	output_name = "./Checkpoints/optimized.pb"
	output_graph_def = optimize_for_inference_lib\
		.optimize_for_inference(
		graph_def, input_node_list,
		output_node_list, tf.float32.as_datatype_enum)
	f = tf.gfile.FastGFile(output_name, "w")
	f.write(output_graph_def.SerializeToString())
	

# Accessing Frozen Models
# Load graph from .pb file
def load_graph():
	frozen_filename = "./Checkpoints/optimized.pb"
	with tf.gfile.GFile(frozen_filename, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(
		graph_def, name="prefix")
	return graph
# Compute network prediction
graph = load_graph()
x = graph.get_tensor_by_name("prefix/x:0")
pred = graph.get_tensor_by_name("prefix/prediction:0")
with tf.Session(graph=graph) as sess:
input_data = np.load("input_filename.npy")
y = sess.run(pred, feed_dict={x: input_data})