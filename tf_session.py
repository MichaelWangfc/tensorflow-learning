#In Tensorflow, we could build and create multiple Tensorflow Sessions using Between-graph Replication for distributed training. 
#MonitoredTrainingSession() coordinates multiple Tensorflow Sessions


summary_hook = tf.train.SummarySaverHook(
	save_steps=100,
	# output_dir=FLAGS.train_dir,
	output_dir=FLAGS.log_dir + '/train',
	summary_op=tf.summary.merge([model.summaries,
								 tf.summary.scalar('Training_Precision', precision)]))

logging_hook = tf.train.LoggingTensorHook(
	tensors={'step': model.global_step,
			 'loss': model.cost,
			 'training precision': precision,
			 'lr': model.lrn_rate},
	every_n_iter=40)

class _LearningRateSetterHook(tf.train.SessionRunHook):
	"""Sets learning_rate based on global step."""

	def begin(self):
		self._lrn_rate = 0.4

	def before_run(self, run_context):
		return tf.train.SessionRunArgs(
			model.global_step,  # Asks for global step value.
			feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

	def after_run(self, run_context, run_values):
		# intel resnet_50_8_nodes version
		train_step = run_values.results

		if train_step < 6240:
			self._lrn_rate = 0.1 + 0.3 * train_step / 6240.0
		elif train_step < 37440:
			self._lrn_rate = 0.4
		elif train_step < 74880:
			self._lrn_rate = 0.1 * 0.4
		elif train_step < 99840:
			self._lrn_rate = 0.01 * 0.4
		else:
			self._lrn_rate = 0.001 * 0.4

config = create_config_proto()
is_chief = (FLAGS.task_index == 0)
with tf.train.MonitoredTrainingSession(
		master=server.target,
		is_chief=is_chief,
		# Loading the ckpt automatically by setting the checkpoint_dir from the latest checkpoint in the directory if any are available
		# An error occurs if any parts of the graph have been modified since the previous checkpoint (in particular variable shapes):
		checkpoint_dir=FLAGS.train_dir, 
		save_checkpoint_steps=1000,
		hooks=[tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
			   logging_hook, _LearningRateSetterHook()],
		chief_only_hooks=[model.replicas_hook, summary_hook],
		# Since we provide a SummarySaverHook, we need to disable default
		# SummarySaverHook. To do that we set save_summaries_steps to 0.
		save_summaries_steps=0,
		stop_grace_period_secs=120,
		# config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
		config=config) as mon_sess:
	while not mon_sess.should_stop():
		mon_sess.run(model.train_op)
		
		
### session and device
tensorflow.python.framework.errors_impl.InvalidArgumentError: Cannot assign a device for operation IteratorToStringHandle: 
Operation was explicitly assigned to /job:worker/task:0/device:GPU:0 but available devices are 
[ /job:localhost/replica:0/task:0/device:CPU:0, /job:localhost/replica:0/task:0/device:GPU:0, /job:localhost/replica:0/task:0/device:XLA_CPU:0, /job:localhost/replica:0/task:0/device:XLA_GPU:0 ]. 
Make sure the device specification refers to a valid device.
