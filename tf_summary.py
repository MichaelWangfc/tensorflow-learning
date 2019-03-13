

#1. we have a new namescope in our graph to hold all the summary ops.
with tf.name_scope("tower_0"):
	tf.summary.scalar("loss",loss)
	tf.summary.histogram(“softmax_w”, softmax_w)
	
#2. merger all the summaries
summary_op = tf.summary.merge_all()

#3.bulit the summary writer to writer into the files
writer=tf.summary.FileWriter(os.getcwd()+'/graph',sess.graph)

#4. Because it’s an op, you have to execute it with sess.run()
loss_batch,_,summary=sess.run([loss,optimizer,summary_op],feed_dict=feed_dict)

#5.you’ve obtained the summary, you need to write the summary to file using the same FileWriter object we created to visual our graph.
writer.add_summary(summary,global_step=step)



###-------------------Example:
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
#get the scalar summary for loss
tf.summary.scalar('loss',loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,label),tf.float32))
#get the scalar summary for accuracy
tf.summary.scalar('accuracy',accuracy)

#merge all the summary
merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
	#bulit the summary writer to writer into the files
	writer = tf.summary.FileWriter(os.getcwd()+'/graph',sess.graph)
	sess.run(init)
	for epoch in range(10):
		cost = 0
		for i in range(len(mini_batches)):
			batch_X,batch_Y = mini_batches[i]
			#execute the summary ops and get the values of summary ops
			_, batch_loss, summary = sess.run([optimizer,loss,merge_summary],feed_dict = {X:batch_X,Y:batch_Y})
			cost += batch_loss/len(mini_batches)
			if i%10==0:
				#add the values of summary into the file
				writer.add_summary(summary,epoch*len(mini_batches)+i)

				
#display both the training and validation accuracy on the same plot in tensorboard. When using tensorboard directly, 
#I have been successful doing something like this (use the same summary object, but different writers).

acc_summary = tf.summary.scalar('accuracy', accuracy)
file_writer1 = tf.summary.FileWriter(logdir+'/train', tf.get_default_graph())
file_writer2 = tf.summary.FileWriter(logdir+'/validation', tf.get_default_graph())
acc_train = acc_summary.eval(feed_dict={X:X_train, y:y_train})
acc_val   = acc_summary.eval(feed_dict={X:X_val, y:y_val})
file_writer1.add_summary(acc_train, epoch)
file_writer2.add_summary(acc_val, epoch)
```				