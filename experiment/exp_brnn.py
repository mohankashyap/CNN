#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 19:06:51
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import csv
import numpy as np
import scipy.io as sio
import theano
import theano.tensor as T
import time
import cPickle
import unittest
from pprint import pprint

sys.path.append('../source/')
from rnn import RNN, TBRNN
from config import RNNConfiger
from wordvec import WordEmbedding
from logistic import SoftmaxLayer


class TestBRNN(unittest.TestCase):
	'''
	Test the performance of BRNN model on sentiment analysis task,
	hope this model will give me good result for a potential publication.
	'''
	def setUp(self):
		'''
		Load training and test texts and labels 
		in sentiment analysis task, preprocessing.
		'''
		np.random.seed(42)
		senti_train_filename = '../data/sentiment-train.txt'
		senti_test_filename = '../data/sentiment-test.txt'
		senti_train_txt, senti_train_label = [], []
		senti_test_txt, senti_test_label = [], []
		start_time = time.time()
		# Read training data set
		with file(senti_train_filename, 'r') as fin:
			reader = csv.reader(fin, delimiter='|')
			for txt, label in reader:
				senti_train_txt.append(txt)
				senti_train_label.append(int(label))
		# Read test data set
		with file(senti_test_filename, 'r') as fin:
			reader = csv.reader(fin, delimiter='|')
			for txt, label in reader:
				senti_test_txt.append(txt)
				senti_test_label.append(int(label))
		end_time = time.time()
		pprint('Time used to load training and test data set: %f seconds.' % (end_time-start_time))
		embedding_filename = '../data/wiki_embeddings.txt'
		# Load training/test data sets and wiki-embeddings
		word_embedding = WordEmbedding(embedding_filename)
		start_time = time.time()
		# Starting and Ending token for each sentence
		self.blank_token = word_embedding.wordvec('</s>')
		pprint('Blank token: ')
		pprint(self.blank_token)
		# Store original text representation
		self.senti_train_txt = senti_train_txt
		self.senti_test_txt = senti_test_txt
		# Word-vector representation
		self.senti_train_label = np.asarray(senti_train_label, dtype=np.int32)
		self.senti_test_label = np.asarray(senti_test_label, dtype=np.int32)
		train_size = len(senti_train_txt)
		test_size = len(senti_test_txt)
		# Check size
		assert train_size == self.senti_train_label.shape[0]
		assert test_size == self.senti_test_label.shape[0]
		pprint('Training size: %d' % train_size)
		pprint('Test size: %d' % test_size)
		# Sequential modeling for each sentence
		self.senti_train_set, self.senti_test_set = [], []
		senti_train_len, senti_test_len = [], []
		# Embedding for training set
		for i, sent in enumerate(senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()))
			vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_train_len.append(len(words)+2)
			self.senti_train_set.append(vectors)
		# Embedding for test set
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()))
			vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_test_len.append(len(words)+2)
			self.senti_test_set.append(vectors)
		assert senti_train_len == [seq.shape[0] for seq in self.senti_train_set]
		assert senti_test_len == [seq.shape[0] for seq in self.senti_test_set]
		end_time = time.time()
		pprint('Time used to build initial training and test matrix: %f seconds.' % (end_time-start_time))
		# Store data
		self.train_size = train_size
		self.test_size = test_size
		self.word_embedding = word_embedding

	# @unittest.skip('Wait a minute')
	def testBRNNonSentiment(self):
		# Set print precision
		np.set_printoptions(threshold=np.nan)

		config_filename = './sentiment_brnn.conf'
		start_time = time.time()
		configer = RNNConfiger(config_filename)
		brnn = TBRNN(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build TBRNN: %f seconds.' % (end_time-start_time))
		n_epoch = 100
		learn_rate = 1e-1
		# Training
		pprint('positive labels: %d' % np.sum(self.senti_train_label))
		pprint('negative labels: %d' % (self.senti_train_label.shape[0]-np.sum(self.senti_train_label)))
		start_time = time.time()
		for i in xrange(n_epoch):
			tot_count = 0
			tot_error = 0.0
			conf_matrix = np.zeros((2, 2), dtype=np.int32)
			tot_grads = np.zeros(brnn.num_params)
			pprint('Total number of parameters in TBRNN: %d' % brnn.num_params)
			for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
				# cost = brnn.train(train_seq, [train_label], learn_rate)
				cost, current_grads = brnn.compute_cost_and_gradient(train_seq, [train_label])
				tot_grads += current_grads
				tot_error += cost
				prediction = brnn.predict(train_seq)[0]
				tot_count += prediction == train_label
				conf_matrix[train_label, prediction] += 1
			# Batch updating 
			tot_grads /= self.train_size
			brnn.update_params(tot_grads, learn_rate)
			accuracy = tot_count / float(self.train_size)
			pprint('Epoch %d, total cost: %f, overall accuracy: %f' % (i, tot_error, accuracy))
			pprint('Confusion matrix: ')
			pprint(conf_matrix)
			# pprint('Gradient vector: ')
			# pprint(tot_grads)
			pprint('-' * 50)
		end_time = time.time()
		pprint('Time used for training: %f minutes.' % ((end_time-start_time)/60))
		# Testing
		tot_count = 0
		for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
			prediction = brnn.predict(test_seq)[0]
			tot_count += test_label == prediction
		pprint('Test accuracy: %f' % (tot_count / float(self.test_size)))
		pprint('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
		pprint('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
		# Re-testing on training set
		tot_count = 0
		for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
			prediction = brnn.predict(train_seq)[0]
			tot_count += train_label == prediction
		pprint('Training accuracy re-testing: %f' % (tot_count / float(self.train_size)))
		# Show representation for training inputs and testing inputs
		start_time = time.time()
		training_forward_rep = np.zeros((self.train_size, configer.num_hidden))
		test_forward_rep = np.zeros((self.test_size, configer.num_hidden))
		training_backward_rep = np.zeros((self.train_size, configer.num_hidden))
		test_backward_rep = np.zeros((self.test_size, configer.num_hidden))
		for i, train_seq in enumerate(self.senti_train_set):
			training_forward_rep[i, :] = brnn.show_forward(train_seq)
			training_backward_rep[i, :] = brnn.show_backward(train_seq)
		for i, test_seq in enumerate(self.senti_test_set):
			test_forward_rep[i, :] = brnn.show_forward(test_seq)
			test_backward_rep[i, :] = brnn.show_backward(test_seq)
		end_time = time.time()
		pprint('Time used to show forward and backward representation for training and test instances: %f seconds' % (end_time-start_time))
		sio.savemat('./BRNN-rep.mat', {'training_forward' : training_forward_rep, 
									   'training_backward' : training_backward_rep, 
									   'test_forward' : test_forward_rep, 
									   'test_backward' : test_backward_rep})
		# Save BRNN
		TBRNN.save('sentiment.brnn.pkl', brnn)
		pprint('Model successfully saved...')

	@unittest.skip('skip')
	def testSoftmaxWithLearned(self):
		data = sio.loadmat('./BRNN-rep.mat')
		train_data = np.hstack((data['training_forward'], data['training_backward']))
		test_data = np.hstack((data['test_forward'], data['test_backward']))
		pprint('Training matrix dimension: ')
		pprint(train_data.shape)
		pprint('Test matrix dimension: ')
		pprint(test_data.shape)
		input = T.matrix(name='input')
		truth = T.ivector(name='label')
		learn_rate = T.scalar(name='learning rate')
		softmax = SoftmaxLayer(input, (100, 2))
		cost = softmax.NLL_loss(truth)
		params = softmax.params
		gradparams = T.grad(cost, params)
		updates = []
		for param, gradparam in zip(params, gradparams):
			updates.append((param, param-learn_rate*gradparam))
		train = theano.function(inputs=[input, truth, learn_rate], outputs=cost, 
								updates=updates)
		nepoch = 20000
		rate = 1e-1
		start_time = time.time()
		for i in xrange(nepoch):
			tot_cost = train(train_data, self.senti_train_label, rate)
			prediction = softmax.predict(train_data)
			accuracy = np.sum(prediction == self.senti_train_label) / float(self.train_size)
			pprint('Epoch %d, total error: %f, accuracy: %f' % (i, tot_cost, accuracy))
		end_time = time.time()
		pprint('Time used for training: %f' % (end_time-start_time))
		# Testing
		prediction = softmax.predict(test_data)
		accuracy = np.sum(prediction == self.senti_test_label) / float(self.test_size)
		pprint('Test accuracy: %f' % accuracy)

	@unittest.skip('bug found!')
	def testResult(self):
		data = sio.loadmat('./BRNN-rep.mat')
		train_data = np.hstack((data['training_forward'], data['training_backward']))
		test_data = np.hstack((data['test_forward'], data['test_backward']))
		pprint('Training matrix dimension: ')
		pprint(train_data.shape)
		pprint('Test matrix dimension: ')
		pprint(test_data.shape)
		start_time = time.time()
		brnn = TBRNN.load('sentiment.brnn.pkl')
		W = brnn.softmax.W.get_value(borrow=True)
		b = brnn.softmax.b.get_value(borrow=True)
		end_time = time.time()
		tmp_softmax = np.dot(train_data, W) + b
		tmp_train_label = np.argmax(tmp_softmax, axis=1)
		accuracy = np.sum(tmp_train_label == self.senti_train_label) / float(self.train_size)
		pprint('Training accuracy: %f' % accuracy)
		tmp_softmax = np.dot(test_data, W) + b
		tmp_test_label = np.argmax(tmp_softmax, axis=1)
		accuracy = np.sum(tmp_test_label == self.senti_test_label) / float(self.test_size)
		pprint('Test accuracy: %f' % accuracy)


if __name__ == '__main__':
	unittest.main()
