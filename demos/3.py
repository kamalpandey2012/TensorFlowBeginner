#reducing the log level
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#program to add nodes
import tensorflow as tf
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
sess = tf.Session()
print(node3)
print(sess.run(node3))