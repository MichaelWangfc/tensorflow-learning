

###===================name_scope名称作用域======================
''''
name_scope 作用于操作，
当然对我们而言还有个更直观的感受就是：在tensorboard 里可视化的时候用名字域进行封装后会更清晰


tensorflow不同位置使用相同的name_scope

name_scope 作用于操作，variable_scope 可以通过设置reuse 标志以及初始化方式来影响域下的变量。
当然对我们而言还有个更直观的感受就是：在tensorboard 里可视化的时候用名字域进行封装后会更清晰

tf.name_scope仅为非tf.get_variable创建的tensor添加前缀；
tf.variable_scope为所有tensor添加前缀
--------------------- 
原文：https://blog.csdn.net/silent56_th/article/details/78231424 
'''

with tf.name_scope('hha'):
    a = tf.zeros(1,name='a')
with tf.name_scope('hha'):
    b = tf.zeros(1,name='b')
print(a.name)
print(b.name)

#返回值为
hha/a:0
hha_1/b:0

#因为tensorflow无法判断是有意在不同位置使用相同的name_scope，而不是两个协作者恰好使用了相同的名字命名不同的name_scope。那tensorflow的做法是对于任意一次调用with tf.name_scope，都自动生成一个独一无二的name_scope以及其名字。所以上诉代码中的a和b所在的name_scope并不相同。 
#那如何强制使得同一个name_scope在不同位置得以应用，上诉网页中给出了方法：在name_scope后面加上’/’，就会使得tensorflow严格使用给定的name_scope作为名字命名当前片段。例如执行如下代码：

with tf.name_scope('hha'):
    a = tf.zeros(1,name='a')
with tf.name_scope('hha/'):
    b = tf.zeros(1,name='b')
print(a.name)
print(b.name)

#返回值为
hha/a:0
hha/b:0


###===================variable_scope变量作用域======================
'''
variable_scope:
原文：
https://blog.csdn.net/zbgjhy88/article/details/78960388 
http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html

目的：
为了实现tensorflow中的变量共享机制：即为了使得在代码的任何部分可以使用某一个已经创建的变量，TF引入了变量共享机制，
使得可以轻松的共享变量，而不用传一个变量的引用。
可以通过设置reuse 标志以及初始化方式来影响域下的变量。



name scope  vs  variable scope
对于使用tf.Variable()方式创建的变量，具有相同的效果，都会在变量名称前面，加上域名称。
对于通过tf.get_variable()方式创建的变量，只有variable scope名称会加到变名称前面，而name scope不会作为前缀。
tf.variable_scope(<scope_name>): 通过 tf.get_variable()为变量名指定命名空间.

原理：
这个方法在建立新的变量时与tf.Variable()完全相同。它的特殊之处在于，他还会搜索是否有同名的变量。
tf.get_variable()方法是TensorFlow提供的比tf.Variable()稍微高级的创建/获取变量的方法，它的工作方式根据当前的变量域（Variable Scope）的
reuse属性变化而变化，我们可以通过tf.get_variable_scope().reuse来查看这个属性，它默认是False。
'''
v = tf.get_variable(name, shape, dtype, initializer)

'''
情况1:当tf.get_variable_scope().reuse == False时，作用域就是为创建新变量所设置的.
这种情况下，v将通过tf.Variable所提供的形状和数据类型来重新创建.创建变量的全称将会由当前变量作用域名+所提供的名字所组成,并且还会检查来
确保没有任何变量使用这个全称.如果这个全称已经有一个变量使用了，那么方法将会抛出ValueError错误.如果一个变量被创建,
他将会用initializer(shape)进行初始化.

情况1：当tf.get_variable_scope().reuse == True时，作用域是为重用变量所设置
这种情况下，调用就会搜索一个已经存在的变量，他的全称和当前变量的作用域名+所提供的名字是否相等.如果不存在相应的变量，
就会抛出ValueError 错误.如果变量找到了，就返回这个变量
'''
tf.variable_scope() 
'''
变量作用域的主方法带有一个名称，它将会作为前缀用于变量名,并且带有一个重用标签来区分以上的两种情况.
嵌套的作用域附加名字所用的规则和文件目录的规则很类似：
'''

tf.get_variable_scope()
#当前变量作用域可以用tf.get_variable_scope()进行检索并且reuse 标签可以通过调用


with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
'''
而tf.variable_scope(scope_name),它会管理在名为scope_name的域（scope）下传递给tf.get_variable的所有变量名（组成了一个变量空间），
根据规则确定这些变量是否进行复用。
这个方法最重要的参数是reuse，有None,tf.AUTO_REUSE与True三个选项。
具体用法如下：
'''

#reuse的默认选项是None,此时会继承父scope的reuse标志。
#自动复用（设置reuse为tf.AUTO_REUSE）,如果变量存在则复用，不存在则创建。这是最安全的用法，在使用新推出的EagerMode时reuse将被强制为tf.AUTO_REUSE选项。用法如下：
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2

#复用（设置reuse=True）：当tf.get_variable_scope().reuse == True时，该方法是为重用变量所设置
with tf.variable_scope("foo"):
  v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
  v1 = tf.get_variable("v", [1])
assert v1 == v

#捕获某一域并设置复用（scope.reuse_variables()）：
with tf.variable_scope("foo") as scope:
  v = tf.get_variable("v", [1])
  scope.reuse_variables()
  v1 = tf.get_variable("v", [1])
assert v1 == v

#1）非复用的scope下再次定义已存在的变量；
#此时调用tf.get_variable(name, shape, dtype, initializer)，我们可以创建一个新的变量（或者说张量），这个变量的名字为name，维度是shape，
#数据类型是dtype，初始化方法是指定的initializer。如果名字为name的变量已经存在的话，会导致ValueError。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    v1 = tf.get_variable("v", [1])
    #  Raises ValueError("... v already exists ...").


#2）定义了复用但无法找到已定义的变量，TensorFlow都会抛出错误，具体如下：
with tf.variable_scope("foo", reuse=True):
    v = tf.get_variable("v", [1])
    #  Raises ValueError("... v does not exists ...").
#该情况下会搜索一个已存在的“foo/v”并将该变量的值赋给v，若找不到“foo/v”变量则会抛出ValueError。

tf.variable_scope(tf.get_variable_scope())


#举例：

w1:sharing variable on CPU
op:execuating the operaiton on each GPU
Traing with 2 GPUS
variable_scope='Conv2d_01', and w1.op.name:Conv2d_01/w1
logits.op.name:tower_0/Fully_conected_01/logits
variable_scope='Conv2d_01', and w1.op.name:Conv2d_01/w1
logits.op.name:tower_1/Fully_conected_01/logits

