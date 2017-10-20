# supressing the logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# program
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
sess = tf.Session()
print(sess.run([node1,node2]))
