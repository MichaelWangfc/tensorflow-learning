#Hello distributed TensorFlow!
'''
To see a simple TensorFlow cluster in action, execute the following:
Start a TensorFlow server as a single-process "cluster".
'''
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
#The tf.train.Server.create_local_server method creates a single-process cluster, with an in-process server.

#Create a cluster
'''
A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. 
Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. 
A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.
To create a cluster, you start one TensorFlow server per task in the cluster. Each task typically runs on a different machine, 
but you can run multiple tasks on the same machine (e.g. to control different GPU devices). In each task, do the following:

1. Create a tf.train.ClusterSpec that describes all of the tasks in the cluster. This should be the same for each task.
2. Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, and identifying the local task with a job name and task index.
'''

#Create a tf.train.ClusterSpec to describe the cluster
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
#Available tasks: /job:local/task:0/job:local/task:1
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
#Available tasks:
/job:worker/task:0
/job:worker/task:1
/job:worker/task:2
/job:ps/task:0
/job:ps/task:1

#Create a tf.train.Server instance in each task
'''
A tf.train.Server object contains a set of local devices, a set of connections to other tasks in its tf.train.ClusterSpec, and a tf.Session that can use these to 
perform a distributed computation. Each server is a member of a specific named job and has a task index within that job. 
A server can communicate with any other server in the cluster.
'''


