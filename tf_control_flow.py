
TensorFlow 提供了几种操作和类用来控制计算图的执行流程和计算条件依赖，我们来具体看看都有哪些方法以及它们的具体含义。

主要有如下计算图流控制方法，我们一个个讲解：
tf.group
tf.cond
tf.identity
tf.tuple
tf.no_op
tf.count_up_to

tf.case
tf.while_loop


###===================tf.group()=========================
#创建一个聚合多个 OP 的 OP ； 当这个 OP 完成后，它输入的所有 OP 也都完成了，相当于一个控制多个 OP 计算进度的功能，这个函数没有返回值。
mul = tf.multiply(w, 2)  
add = tf.add(w, 2)  
group = tf.group(mul, add)  
print sess.run(group)  

#或者：
update_ops = []
update_ops.append(grad_updates)
update_op = tf.group(*update_ops)

#简而言之，这是一个聚合多个操作的操作，从示例中可以看出，它的主要用途是更加简便的完成多个操作。

###===================tf.cond()================
#这是一个根据条件进行流程控制的函数，它表现出像python中if...else...相似的功能。
#它有三个参数，pred, true_fn,false_fn ,它的主要作用是在 pred 为真的时候返回 true_fn 函数的结果，为假的时候返回 false_fn 函数的结果，我们通过一个示例来看：

scale = tf.cond(tf.greater(height, width),
                lambda: x / y,
                lambda: y / x )
#如果 height 更大，就执行 x / y，反之则是 y / x .

###===================tf.where()=========
#在使用TensorFlow的大部分时间中，我们都会使用大型的张量，而且想要在一个批次（a batch）中进行操作。一个与之相关的条件操作符是tf.where()，
#它需要提供一个条件判断，就和tf.cond()一样，但是tf.where()将会根据这个条件判断，在一个批次中选择输出，如：

a = tf.constant([1, 1])
b = tf.constant([2, 2])
p = tf.constant([True, False])
x = tf.where(p, a + b, a * b)
print(tf.Session().run(x))
输出[3, 2]

###===================tf.identity()================
那么在什么时候使用这个方法呢？

它是通过在计算图内部创建 send / recv节点来引用或复制变量的，最主要的用途就是更好的控制在不同设备间传递变量的值；



另外，它还有一种常见的用途，就是用来作为一个虚拟节点来控制流程操作，比如我们希望强制先执行loss_averages_op或updata_op，然后更新相关变量。这可以实现为：

with tf.control_dependencies（[loss_averages_op]）：
  total_loss = tf.identity（total_loss）
或者：

with tf.control_dependencies（[updata_op]）：
  train_tensor = tf.identity（total_loss,name='train_op'）
在这里，tf.identity除了在执行 loss_averages_op之后标记total_loss张量被执行之外没有做任何有用的事情。