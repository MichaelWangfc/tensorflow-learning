
tf.train

tf.train.exponential_decay(
    learning_rate,
    global_step,
    decay_steps,
    decay_rate,
    staircase=False,
    name=None
)

#The function returns the decayed learning rate. It is computed as:
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)


# Decay the learning rate exponentially based on the number of steps.
lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
								global_step,
								decay_steps,
								cifar10.LEARNING_RATE_DECAY_FACTOR,
								staircase=True)

# Create an optimizer that performs gradient descent.
opt = tf.train.GradientDescentOptimizer(lr)



tf.train.AdamOptimizer
