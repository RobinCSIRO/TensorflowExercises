#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:22:53 2018

@author: wan246
"""

import tensorflow as tf
hello = tf.constant('hello')
sess = tf.Session()
print(sess.run(hello))