#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-05 16:52:45
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import cPickle
import time
import unittest
from pprint import pprint
import numpy as np
import theano
import theano.tensor as T
sys.path.append('../source/')

from utils import floatX
from logistic import SoftmaxLayer
from wordvec import WordEmbedding
from mlp import MLP
from config import MLPConfiger

class TestSnippet(unittest.TestCase):
	def setUp(self):
		'''
		Load and store the snippet data set.
		'''
		np.random.seed(42)
		snippet_train_set_filename = '../data/train_snip.txt'
		snippet_test_set_filename = '../data/test_snip.txt'
		snippet_train_label_filename = '../data/train_label.txt'
		snippet_test_label_filename = '../data/test_label.txt'
		embedding_filename = '../data/wiki_embeddings.txt'
		# Build architecture of CNN from the configuration file
		# Load wiki-embedding
		word_embedding = WordEmbedding(embedding_filename)
		# Load data and train via minibatch
		with file(snippet_train_set_filename, 'rb') as fin:
			snippet_train_txt = fin.readlines()
		with file(snippet_test_set_filename, 'rb') as fin:
			snippet_test_txt = fin.readlines()
		snippet_train_label = np.loadtxt(snippet_train_label_filename, dtype=np.int32)
		snippet_test_label = np.loadtxt(snippet_test_label_filename, dtype=np.int32)
		training_size = len(snippet_train_txt)
		test_size = len(snippet_test_txt)
		# Check size:
		pprint('Training size: %d' % training_size)
		pprint('Test size: %d' % test_size)
		assert training_size == snippet_train_label.shape[0]
		assert test_size == snippet_test_label.shape[0]
		# Word embedding
		snippet_train_set = np.zeros((training_size, word_embedding.embedding_dim()), dtype=floatX)
		snippet_test_set = np.zeros((test_size, word_embedding.embedding_dim()), dtype=floatX)

		for i, snippet in enumerate(snippet_train_txt):
			words = snippet.split()
			vectors = np.asarray([word_embedding.wordvec(word) for word in words], dtype=floatX)
			snippet_train_set[i, :] = np.mean(vectors, axis=0)

		for i, snippet in enumerate(snippet_test_txt):
			words = snippet.split()
			vectors = np.asarray([word_embedding.wordvec(word) for word in words], dtype=floatX)
			snippet_test_set[i, :] = np.mean(vectors, axis=0)
		# Shuffle training and test data set
		train_rand_index = np.random.permutation(training_size)
		test_rand_index = np.random.permutation(test_size)
		snippet_train_set = snippet_train_set[train_rand_index, :]
		snippet_train_label = snippet_train_label[train_rand_index]
		snippet_test_set = snippet_test_set[test_rand_index, :]
		snippet_test_label = snippet_test_label[test_rand_index]
		# Decrease 1 from label
		snippet_train_label -= 1
		snippet_test_label -= 1
		self.snippet_train_set = snippet_train_set
		self.snippet_train_label = snippet_train_label
		self.snippet_test_set = snippet_test_set
		self.snippet_test_label = snippet_test_label

	@unittest.skip('Accuracy: 0.8513')
	def testMLP(self):
		'''
		Using MLP of one hidden layer and one softmax layer
		'''
		conf_filename = './snippet_mlp.conf'
		start_time = time.time()
		configer = MLPConfiger(conf_filename)
		mlpnet = MLP(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build the architecture of MLP: %f seconds' % (end_time-start_time))
		# Training
		start_time = time.time()
		for i in xrange(configer.nepoch):
			cost, accuracy = mlpnet.train(self.snippet_train_set, self.snippet_train_label)
			pprint('epoch %d, cost = %f, accuracy = %f' % (i, cost, accuracy))
		end_time = time.time()
		pprint('Time used for training MLP network on Snippet task: %f minutes' % ((end_time-start_time)/60))
		# Test
		test_size = self.snippet_test_label.shape[0]
		prediction = mlpnet.predict(self.snippet_test_set)
		accuracy = np.sum(prediction == self.snippet_test_label) / float(test_size)
		pprint('Test accuracy: %f' % accuracy)

	@unittest.skip('Accuracy: 0.8535')
	def testSoftmax(self):
		'''
		Using Softmax classifier
		'''
		input = T.matrix(name='input')
		truth = T.ivector(name='label')
		learning_rate = T.scalar(name='learning rate')
		num_in, num_out = 50, 8
		softmax = SoftmaxLayer(input, (num_in, num_out))
		lambdas = 1e-5
		cost = softmax.NLL_loss(truth) + lambdas * softmax.L2_loss()
		params = softmax.params
		gradparams = T.grad(cost, params)
		updates = []
		for param, gradparam in zip(params, gradparams):
			updates.append((param, param-learning_rate*gradparam))
		obj = theano.function(inputs=[input, truth, learning_rate], outputs=cost, updates=updates)
		# Training
		nepoch = 5000
		start_time = time.time()
		for i in xrange(nepoch):
			rate = 2.0 / (1.0 + i/500)
			func_value = obj(self.snippet_train_set, self.snippet_train_label, rate)		
			prediction = softmax.predict(self.snippet_train_set)
			accuracy = np.sum(prediction == self.snippet_train_label) / float(self.snippet_train_label.shape[0])
			pprint('epoch %d, cost = %f, accuracy = %f' % (i, func_value, accuracy))
		end_time = time.time()
		pprint('Time used to train the softmax classifier: %f minutes.' % ((end_time-start_time)/60))
		# Test
		test_size = self.snippet_test_label.shape[0]
		prediction = softmax.predict(self.snippet_test_set)
		accuracy = np.sum(prediction == self.snippet_test_label) / float(test_size)
		pprint('Test accuracy: %f' % accuracy)

	def testCNN(self):
		pass

if __name__ == '__main__':
	unittest.main()








