# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:41:35 2018

@author: wangg
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

LR = 1e-3
training_epochs = 500
display_step = 10
batch_size = 128
dropout = 0.75

Weights = {
        "Conv1": tf.Variable(tf.random_normal([5,5,1,32])),
        "Conv2": tf.Variable(tf.random_normal([5,5,32,64])),
        "FC1":   tf.Variable(tf.random_normal([7*7*64,1024])),
        "FC2":   tf.Variable(tf.random_normal([1024,10]))
        }

Biases = {
        "bc1": tf.Variable(tf.random_normal([32])),
        "bc2": tf.Variable(tf.random_normal([64])),
        "bf1":   tf.Variable(tf.random_normal([1024])),
        "bf2":   tf.Variable(tf.random_normal([10]))
        }

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def ConvMNIST(x,weights,biases,dropout):
    x = tf.reshape(x,shape=[-1,28,28,1])
    conv1 = conv2d(x,weights["Conv1"],biases["bc1"])
    conv1 = maxpool2d(conv1,k=2)
    conv2 = conv2d(conv1,weights["Conv2"],biases["bc2"])
    conv2 = maxpool2d(conv2,k=2)
    fc1 = tf.reshape(conv2, [-1, weights["FC1"].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1,weights["FC1"]),biases["bf1"])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)
    fc2 = tf.add(tf.matmul(fc1,weights["FC2"]),biases["bf2"])
    return fc2

logits = ConvMNIST(X,Weights,Biases,keep_prob)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()
#
with tf.Session() as sess:
    sess.run(init)
    for epochs in range(1,training_epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train,feed_dict={X:batch_x,Y:batch_y,keep_prob: dropout})
#        print('finished')
        if epochs%display_step ==0 or epochs ==1:
            loss = sess.run(loss_op, feed_dict={X:batch_x,Y:batch_y,keep_prob: 1.0})
            acc = sess.run(accuracy, feed_dict={X:batch_x,Y:batch_y,keep_prob: 1.0})
            print("Step " + str(epochs) + ", Minibatch Loss= " +"{:.4f}".format(loss)+",accuracy="\
                  +"{:.3f}".format(acc))
#            
    print('Training Finished!')
    test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels,keep_prob: 1.0})
    print("Testing result:"+"{:.3f}".format(test_acc))