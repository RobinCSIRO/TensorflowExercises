# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:47:38 2018

@author: wangg
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder('float',[None,10])

LR = 1e-2
training_epochs = 5000
display_step = 100
batch_size = 128

n_hidden1 = 256
n_hidden2 = 128
n_input = 784
n_output = 10

Weights = {
        'h1': tf.Variable(tf.random_normal([n_input,n_hidden1])),
        'h2': tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
        'out': tf.Variable(tf.random_normal([n_hidden2,n_output]))
        }
Biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden1])),
        'h2': tf.Variable(tf.random_normal([n_hidden2])),
        'out': tf.Variable(tf.random_normal([n_output]))
        }

def MLP(x):
    layer1 = tf.add(tf.matmul(x,Weights['h1']),Biases['h1'])
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(layer1,Weights['h2']),Biases['h2'])
    layer2 = tf.nn.relu(layer2)
    output = tf.add(tf.matmul(layer2,Weights['out']),Biases['out'])
    return output


logits = MLP(X)
pred = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
train = tf.train.AdamOptimizer(learning_rate = LR).minimize(loss_op)
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epochs in range(1,training_epochs+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train,feed_dict={X:batch_x,Y:batch_y})
        if epochs%display_step ==0 or epochs ==1:
            loss = sess.run(loss_op, feed_dict={X:batch_x,Y:batch_y})
            acc = sess.run(accuracy, feed_dict={X:batch_x,Y:batch_y})
            print("Step " + str(epochs) + ", Minibatch Loss= " +"{:.4f}".format(loss)+",accuracy="\
                  +"{:.3f}".format(acc))
            
    print('Training Finished!')
    test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images,Y: mnist.test.labels})
    print("Testing result:"+"{:.3f}".format(test_acc))
