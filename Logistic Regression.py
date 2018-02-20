# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:39:30 2018

@author: wangg
"""

#Logistic Regression with MNIST#

import tensorflow as tf
#import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])

learning_rate = 1e-2
training_epoch = 30
batch_size = 100
display_step = 1

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(X,W)+b)
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([train,loss],feed_dict={X:batch_x,Y:batch_y})
            avg_cost += c / total_batch
        if (i+1) % display_step == 0:
            print("epochs:",'%04d'% (i+1), "loss=","{:.9f}".format(avg_cost))
            
    print("Training Finished!")
    
    correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("Test_accuracy=",accuracy.eval({X: mnist.test.images[:3000], Y: mnist.test.labels[:3000]}))
