# Tensorflow tutorials
## Introduction
The central unit of data in TensorFlow is the tensor. A tensor consists of a set of primitive values shaped into an array of any number of dimensions

## Importing Tensor flow
```
import tensorflow as tf
```
This gives Python access to all of TensorFlow's classes, methods, and symbols

## The computational graph
Tensorflow core program consist of two sections
1. Building the computational graph 
2. Running the computational graph

A **computational graph** is a series of TensorFlow operations arranged into graph of nodes
A constant is a type of node that takes no input and output a value stored internally
```
//1.py
import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) //implicitly tf.float32
print(node1, node2)
```
output
```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```
The output of tensor doesn't give values instead the nodes that could be evaluated to output 3.0 and 4.0

To actually evaluate the nodes we must evaluate within TensorFlow **session**

```
//2.py
session = tf.Session()
print(session.run([node1, node2]))
```
now we can expect values 3.0 and 4.0

More advance computations could be performed using tensorflow operations
```
#3.py
node3 = tf.add(node1, node2)
print('node3',node3)
print('session.run(node3)',session.run(node3))
```

From above example it is clear that graph is of not very much importance as it takes constant input. Parameterising that input and allowing it to take external inputs is named as **placeholders**. A **placeholder** is a promise to provide a value later

```
#4.py
a = tf.placeholder(tf.float32)
b= tf.placeholder(tf.float32)
adder_node = a + b # +: shortcut of tf.add()
print(sess.run(adder_node, {a:3, b:4.5}))
print(sess.run(adder_node, {a:[1,3],b:[2,4]}))
```
we can make computational graph more complex by adding more operations
```
$4.py
add_and_triple = adder_node*3
print(sess.run(add_and_triple, {a:3, b:4.5}))
```
In machine learning we will want a model that takes arbitary inputs. To make the model trainable, we need to be able to modify graph to get new outputs with the same input. **Variables** allow us to add trainable parameters to the graph.

```
M = tf.Variable([.3], dtype=tf.float32)
c = tf.Varialbe([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = M*x + c;
```
constants get initialized using tf.constants but variables cannot be initialized using tf.Variable. To initialize variables we need to explicitly call a special operation as follows
```
init = tf.global_variables_initializer()
sess.run(init)
```
to evaluate the graph
```
print(sess.run(linear_model, {x: [1,2,3,4]}))
```
to produce the output
```
[ 0.          0.30000001  0.60000002  0.90000004]
```
We have created the model that we don't know how good is it. To evaluate a model on training data we need a y placeholder to provide the desired values, we need to write a loss function

The **loss function** tells how far apart the trained model is from provided data. We will use a standard loss model for linear regression, which sums the square of deltas between the current model and the provided data.
```
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y: [0, -1, -2, -3]}))
```
producing the loss value 
```
23.66
```
we could reassign values the values of w and b to the perfect values of -1 and 1. The values could be changed using tf.assign 
```
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1,-2,-3]}))
```
the final print now shows that the loss is now zero

