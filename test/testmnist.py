#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 17:54:38
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
sys.path.append('../source/')
import cPickle
import unittest
import numpy as np
from pprint import pprint
from cnn import ConvNet

class TestMNIST(unittest.TestCase):
	def setUp(self):
		fname = '../experiment/mnist.cnn'
		self.convnet = ConvNet.load(fname)		
		mnist_filename = '../data/mnist.pkl'
		fin = file(mnist_filename, 'rb')
		tr, va, te = cPickle.load(fin)
		fin.close()
		training_set = np.vstack((tr[0], va[0]))
		training_label = np.hstack((tr[1], va[1]))
		test_set, test_label = te[0], te[1]
		training_size = training_set.shape[0]
		test_size = test_set.shape[0]
		# Convert data type into int32
		training_label = training_label.astype(np.int32)
		test_label = test_label.astype(np.int32)
		# Check
		pprint('Dimension of Training data set: (%d, %d)' % training_set.shape)
		pprint('Dimension of Test data set: (%d, %d)' % test_set.shape)
		# Shuffle
		train_rand_shuffle = np.random.permutation(training_size)
		test_rand_shuffle = np.random.permutation(test_size)
		training_set = training_set[train_rand_shuffle, :]
		training_label = training_label[train_rand_shuffle]
		test_set = test_set[test_rand_shuffle, :]
		test_label = test_label[test_rand_shuffle]
		self.test_set = test_set.reshape((10000, 1, 28, 28))
		self.test_label = test_label

	def testPerformance(self):
		batch_size = 500
		image_row, image_col = 28, 28
		test_size = self.test_set.shape[0]
		test_set = self.test_set
		test_label = self.test_label
		num_batches = test_size / batch_size
		right_count = 0
		for i in xrange(num_batches):
			minibatch = test_set[i*batch_size : (i+1)*batch_size]
			label = test_label[i*batch_size : (i+1)*batch_size]
			minibatch = minibatch.reshape((batch_size, 1, image_row, image_col))
			prediction = self.convnet.predict(minibatch)
			right_count += np.sum(prediction == label)
		test_accuracy = right_count / float(test_size)
		pprint('Test set accuracy: %f' % test_accuracy)


if __name__ == '__main__':
	unittest.main()



