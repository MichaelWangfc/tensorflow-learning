'''
class tf.train.Optimizer
--------------------- 
原文：https://blog.csdn.net/lenbow/article/details/52218551 

该模块定义了一个训练模型的operator的API。一般来说不会直接用到这个API，它更像是一个父类，各种基于不同类型优化算法的optimizer都继承自这里，
如：GradientDescentOptimizer, AdagradOptimizer, MomentumOptimizer。
'''


'''
minimize()操作

对optimizer调用minimizer()函数,获取所有trainable_variables的梯度包含两步：
1. 计算梯度
optimizer.compute_gradients()
2. 将梯度应用到参数更新中
optimizer.apply_gradients()

使用minimize()操作，该操作不仅可以优化更新训练的模型参数，也可以为全局步骤(global step)计数。
与其他tensorflow操作类似，这些训练操作都需要在tf.session会话中进行
'''

global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
