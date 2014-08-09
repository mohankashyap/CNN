#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-08 14:26:38
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
import scipy.io as sio
import matplotlib.pyplot as plt
import PIL
import imgutils

from pprint import pprint
from theano.tensor.shared_randomstreams import RandomStreams
from mlp import AutoEncoder
from activations import Activation

class TestAE(unittest.TestCase):
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

	@unittest.skip('Model been trained, finished...')
	def testAE(self):
		# Set parameters
		input = T.matrix(name='input')
		num_in, num_out = 784, 500
		act = Activation('sigmoid')
		is_denoising, is_sparse = True, False
		lambda1 = 1e-4
		mask = 0.7
		rng = RandomStreams(42)
		start_time = time.time()
		ae = AutoEncoder(input, (num_in, num_out), act, is_denoising, is_sparse, 
						lambda1, mask, rng, verbose=True)
		end_time = time.time()
		pprint('Time used to build the AutoEncoder: %f seconds.' % (end_time-start_time))
		batch_size = 1000
		num_batches = self.training_set.shape[0] / batch_size
		nepoch = 50
		learn_rate = 1
		start_time = time.time()
		for i in xrange(nepoch):
			rate = learn_rate
			for j in xrange(num_batches):
				train_set = self.training_set[j*batch_size : (j+1)*batch_size, :]
				cost = ae.train(train_set, rate)
				pprint('epoch %d, batch %d, cost = %f' % (i, j, cost))
		end_time = time.time()
		pprint('Time used for training AutoEncoder: %f seconds.' % (end_time-start_time))
		image = PIL.Image.fromarray(imgutils.tile_raster_images(
						X=ae.encode_layer.W.get_value(borrow=True).T,
						img_shape=(28, 28), tile_shape=(10, 10),
						tile_spacing=(1, 1)))
		image.save('filters_corruption_%.2f.png' % mask)
		AutoEncoder.save('./autoencoder-mnist.model', ae)

	@unittest.skip('Not ready yet...')
	def testRecons(self):
		'''
		Test the compression and reconstruction performance 
		of the learned denoising auto-encoder.
		'''
		fname = './autoencoder-mnist.model'
		ae = AutoEncoder.load(fname)
		sio.savemat('test_set.mat', {'data' : self.test_set})
		compressed_data = ae.compress(self.test_set)
		sio.savemat('compressed_data.mat', {'data' : compressed_data})
		reconstructed_data = ae.reconstruct(self.test_set)
		sio.savemat('reconstructed_data.mat', {'data' : reconstructed_data})
		pprint('Save all data into matlab version, finished...')

if __name__ == '__main__':
	unittest.main()