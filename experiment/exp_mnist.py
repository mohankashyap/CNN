#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-05 16:52:45
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import cPickle
import time
from pprint import pprint
import numpy as np
sys.path.append('../source/')

from cnn import ConvNet
from config import CNNConfiger

np.random.seed(42)
mnist_filename = '../data/mnist.pkl'
conf_filename = './mnist.conf'
# Build architecture of CNN from the configuration file
start_time = time.time()
configer = CNNConfiger(conf_filename)
convnet = ConvNet(configer, verbose=True)
end_time = time.time()
pprint('Time used to build the architecture of CNN: %f seconds' % (end_time-start_time))
# Load data and train via minibatch
fin = file(mnist_filename, 'rb')
tr, va, te = cPickle.load(fin)
fin.close()
training_set = np.vstack((tr[0], va[0]))
training_label = np.hstack((tr[1], va[1]))
test_set, test_label = te[0], te[1]
training_size = training_set.shape[0]
test_size = test_set.shape[0]
# Check
pprint('Dimension of Training data set: (%d, %d)' % training_set.shape)
pprint('Dimension of Test data set: (%d, %d)' % test_set.shape)
# Shuffle
train_rand_shuffle = np.random.permutation(train_size)
test_rand_shuffle = np.random.permutation(test_size)
training_set = training_set[train_rand_shuffle, :]
training_label = training_label[train_rand_shuffle]
test_set = test_set[test_rand_shuffle, :]
test_label = test_label[test_rand_shuffle]
# Partition data based on batch size
batch_size = configer.batch_size
nepoch = configer.nepoch
num_batches = training_size / batch_size
start_time = time.time()
for i in xrange(nepoch):
	for j in xrange(num_batches):
		minibatch = training_set[j*batch_size : (j+1)*batch_size, :]
		label = training_label[j*batch_size : (j+1)*batch_size]
		cost, accuracy = convnet.train(minibatch, label)
		pprint('Epoch %d, batch %d, cost = %f, accuracy = %f' % (i, j, cost, accuracy))
	ConvNet.save('./mnist.cnn', convnet)
end_time = time.time()
pprint('Time used to train CNN on MNIST: %f minutes' % (end_time-start_time) / 60)







