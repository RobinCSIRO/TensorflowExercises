# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:55:30 2018

@author: wangg
"""
#Single variable Linear Regression#
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

train_x = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
x = tf.placeholder("float")
y = tf.placeholder('float')
n_samples = train_x.shape[0]

learning_rate = 1e-2
training_epoch = 1000
display_step = 50

W = tf.Variable(rng.randn(),name = 'weight')
b = tf.Variable(rng.randn(),name = 'biases')

pred = tf.add(tf.multiply(W,x),b)
loss = tf.reduce_sum(tf.pow(pred-y,2))/(2*n_samples)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epoch):
        for (X,Y) in zip (train_x,train_y):
            sess.run(train,feed_dict={x:X,y:Y})
            
        if (i+1)%display_step == 0:
            c = sess.run(loss,feed_dict={x:X,y:Y})
            print("Epoch:",'%04d' % (i+1), "cost=","{:.09f}".format(c),"W=",sess.run(W),"b=",sess.run(b))
            
    print('optimization finished!')
    cost = sess.run(loss,feed_dict={x:X,y:Y})
    print("training_loss=",cost, "W=", sess.run(W), "b=", sess.run(b))
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    
#Testing Part#
    test_x = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    test_loss = sess.run(tf.reduce_sum(tf.pow(pred-y,2))/(2*test_x.shape[0]),feed_dict={x:test_x,y:test_y})
    print("test_loss=",test_loss)
    plt.plot(test_x,test_y,'ro',label='test data')
    plt.plot(test_x,sess.run(W)*test_x+sess.run(b),label = 'Fitted line')
    plt.legend()
    plt.show()
