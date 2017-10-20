#reducing the log noise
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#program to add two nodes parameterized
import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b
sess = tf.Session()
print(sess.run(adder_node,{ a: 3, b: 4.5}))
print(sess.run(adder_node, {a:[1,3],b:[2,4]}))

#more operations
add_and_triple = adder_node*3
print('triple', sess.run(add_and_triple,{a:3,b:4.5}))
