#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-08 19:41:23
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import cPickle
sys.path.append('../source/')
import theano
import theano.tensor as T
import numpy as np
import unittest
import time

from pprint import pprint
from theano.tensor.shared_randomstreams import RandomStreams
from config import DAEConfiger
from mlp import AutoEncoder
from mlp import DAE
from activations import Activation


class TestDAE(unittest.TestCase):
	def setUp(self):
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
		# Store for later use
		self.training_set = training_set
		self.test_set = test_set
		self.training_label = training_label
		self.test_label = test_label

	# @unittest.skip('Model been trained, finished...')
	def testDAE(self):
		# Set parameters
		configer = DAEConfiger('../experiment/mnist_dae.conf')
		start_time = time.time()
		dae = DAE(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build the architecture of DAE: %f seconds.' % (end_time-start_time))
		batch_size = 1000
		num_batches = self.training_set.shape[0] / batch_size
		learn_rate = 1
		start_time = time.time()
		for i in xrange(configer.nepoch):
			rate = learn_rate
			for j in xrange(num_batches):
				cost = dae.train(self.training_set[j*batch_size : (j+1)*batch_size, :], learn_rate)
				pprint('epoch %d, batch %d, cost = %f' % (i, j, cost))
			DAE.save('dae-mnist.model', dae)
		end_time = time.time()
		pprint('Time used for training Deep Auto-Encoder: %f minutes.' % ((end_time-start_time)/60))
		pprint('Model save finished...')



if __name__ == '__main__':
	unittest.main()
