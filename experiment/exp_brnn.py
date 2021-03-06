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
import logging
from pprint import pprint

sys.path.append('../source/')
from rnn import RNN, TBRNN, BRNN
from config import RNNConfiger
from wordvec import WordEmbedding
from logistic import SoftmaxLayer
from utils import floatX

theano.config.openmp=True
theano.config.exception_verbosity='high'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

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
		# Record id of words for fine-tuning
		senti_train_words_label, senti_test_words_label = [], []
		# Load Word-Embedding
		embedding_filename = '../data/wiki_embeddings.txt'
		# Load training/test data sets and wiki-embeddings
		word_embedding = WordEmbedding(embedding_filename)
		# Starting and Ending token for each sentence
		self.blank_token = word_embedding.wordvec('</s>')
		self.blank_index = word_embedding.word2index('</s>')
		# Read training data set
		with file(senti_train_filename, 'r') as fin:
			reader = csv.reader(fin, delimiter='|')
			for txt, label in reader:
				senti_train_txt.append(txt)
				senti_train_label.append(int(label))
				words = txt.split()
				words = [word.lower() for word in words]
				tmp_indices = np.zeros(len(words)+2, dtype=np.int32)
				tmp_indices[0] = self.blank_index
				tmp_indices[1:-1] = np.asarray([word_embedding.word2index(word) for word in words])
				tmp_indices[-1] = self.blank_index
				senti_train_words_label.append(tmp_indices)
		# Read test data set
		with file(senti_test_filename, 'r') as fin:
			reader = csv.reader(fin, delimiter='|')
			for txt, label in reader:
				senti_test_txt.append(txt)
				senti_test_label.append(int(label))
				words = txt.split()
				words = [word.lower() for word in words]
				tmp_indices = np.zeros(len(words)+2, dtype=np.int32)
				tmp_indices[0] = self.blank_index
				tmp_indices[1:-1] = np.asarray([word_embedding.word2index(word) for word in words])
				tmp_indices[-1] = self.blank_index
				senti_test_words_label.append(tmp_indices)
		end_time = time.time()
		logger.debug('Time used to load training and test data set: %f seconds.' % (end_time-start_time))
		start_time = time.time()
		# Store original word index representation
		self.senti_train_words_label = senti_train_words_label
		self.senti_test_words_label = senti_test_words_label
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
		logger.debug('Training size: %d' % train_size)
		logger.debug('Test size: %d' % test_size)
		# Sequential modeling for each sentence
		self.senti_train_set, self.senti_test_set = [], []
		senti_train_len, senti_test_len = [], []
		# Embedding for training set
		for i, sent in enumerate(senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
			vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_train_len.append(len(words)+2)
			self.senti_train_set.append(vectors)
		# Embedding for test set
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
			vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_test_len.append(len(words)+2)
			self.senti_test_set.append(vectors)
		assert senti_train_len == [seq.shape[0] for seq in self.senti_train_set]
		assert senti_test_len == [seq.shape[0] for seq in self.senti_test_set]
		end_time = time.time()
		logger.debug('Time used to build initial training and test matrix: %f seconds.' % (end_time-start_time))
		# Store data
		self.train_size = train_size
		self.test_size = test_size
		self.word_embedding = word_embedding

	@unittest.skip('Wait a minute')
	def testTBRNNonSentiment(self):
		# Set print precision
		np.set_printoptions(threshold=np.nan)

		config_filename = './sentiment_brnn.conf'
		start_time = time.time()
		configer = RNNConfiger(config_filename)
		brnn = TBRNN(configer, verbose=True)
		# brnn = TBRNN.load('sentiment.brnn.Sep5.pkl')
		end_time = time.time()
		logger.debug('Time used to load TBRNN: %f seconds.' % (end_time-start_time))
		# Training
		logger.debug('positive labels: %d' % np.sum(self.senti_train_label))
		logger.debug('negative labels: %d' % (self.senti_train_label.shape[0]-np.sum(self.senti_train_label)))
		start_time = time.time()
		## AdaGrad learning algorithm instead of the stochastic gradient descent algorithm
		history_grads = np.zeros(brnn.num_params)
		n_epoch = 2000
		learn_rate = 1
		fudge_factor = 1e-6
		for i in xrange(n_epoch):
			tot_count = 0
			tot_error = 0.0
			conf_matrix = np.zeros((2, 2), dtype=np.int32)
			tot_grads = np.zeros(brnn.num_params)
			logger.debug('Total number of parameters in TBRNN: %d' % brnn.num_params)
			for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
				# cost = brnn.train(train_seq, [train_label], learn_rate)
				cost, current_grads = brnn.compute_cost_and_gradient(train_seq, [train_label])
				tot_grads += current_grads
				tot_error += cost
				# historical gradient accumulation
				history_grads += current_grads ** 2
				# predict current training label
				prediction = brnn.predict(train_seq)[0]
				tot_count += prediction == train_label
				conf_matrix[train_label, prediction] += 1
			# Batch updating 
			tot_grads /= self.train_size
			# Update historical gradient vector
			adjusted_grads = tot_grads / (fudge_factor + np.sqrt(history_grads))
			brnn.update_params(adjusted_grads, learn_rate)
			# End of the core AdaGrad updating algorithm
			accuracy = tot_count / float(self.train_size)
			logger.debug('Epoch %d, total cost: %f, overall accuracy: %f' % (i, tot_error, accuracy))
			logger.debug('Confusion matrix: ')
			logger.debug(conf_matrix)
			logger.debug('-' * 50)
			if (i+1) % 100 == 0:
				logger.debug('=' * 50)
				logger.debug('Test at epoch: %d' % i)
				# Testing
				tot_count = 0
				for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
					prediction = brnn.predict(test_seq)[0]
					tot_count += test_label == prediction
				logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
				logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
				logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
				logger.debug('=' * 50)
		end_time = time.time()
		logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
		# Testing
		tot_count = 0
		for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
			prediction = brnn.predict(test_seq)[0]
			tot_count += test_label == prediction
		logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
		logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
		logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
		# Re-testing on training set
		tot_count = 0
		for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
			prediction = brnn.predict(train_seq)[0]
			tot_count += train_label == prediction
		logger.debug('Training accuracy re-testing: %f' % (tot_count / float(self.train_size)))
		# Show representation for training inputs and testing inputs
		start_time = time.time()
		training_forward_rep = np.zeros((self.train_size, configer.num_hidden))
		test_forward_rep = np.zeros((self.test_size, configer.num_hidden))
		training_backward_rep = np.zeros((self.train_size, configer.num_hidden))
		test_backward_rep = np.zeros((self.test_size, configer.num_hidden))
		# Save TBRNN
		TBRNN.save('sentiment.brnn.Sep5_2.pkl', brnn)
		logger.debug('Model successfully saved...')

	# @unittest.skip('Wait a minute')
	def testBRNNonSentiment(self):
		# Set print precision
		np.set_printoptions(threshold=np.nan)

		config_filename = './sentiment_brnn.conf'
		start_time = time.time()
		configer = RNNConfiger(config_filename)
		brnn = BRNN.load('./sentiment.nbrnn.Oct.pkl')
		# brnn = BRNN(configer, verbose=True)
		end_time = time.time()
		logger.debug('Time used to build BRNN: %f seconds.' % (end_time-start_time))
		# Training
		logger.debug('positive labels: %d' % np.sum(self.senti_train_label))
		logger.debug('negative labels: %d' % (self.senti_train_label.shape[0]-np.sum(self.senti_train_label)))
		start_time = time.time()
		## AdaGrad learning algorithm instead of the stochastic gradient descent algorithm
		# history_grads = np.zeros(brnn.num_params, dtype=floatX)
		n_epoch = 2000
		learn_rate = 0.1
		fudge_factor = 1e-6
		batch_size = 200
		for i in xrange(n_epoch):
			learn_rate = 0.1 / (1 + i/100)
			tot_count = 0
			tot_error = 0.0
			conf_matrix = np.zeros((2, 2), dtype=np.int32)
			tot_grads = np.zeros(brnn.num_params, dtype=floatX)
			for j, (train_seq, train_label) in enumerate(zip(self.senti_train_set, self.senti_train_label)):
				# Reach a pre-specified batch size, updating the gradients
				if (j+1) % batch_size == 0:
					tot_grads /= batch_size
					brnn.update_params(tot_grads, learn_rate)
					tot_grads = np.zeros(brnn.num_params, dtype=floatX)
				cost, current_grads = brnn.compute_cost_and_gradient(train_seq, [train_label])
				tot_grads += current_grads
				tot_error += cost
				# historical gradient accumulation
				# history_grads += current_grads ** 2
				# predict current training label
				prediction = brnn.predict(train_seq)[0]
				tot_count += prediction == train_label
				conf_matrix[train_label, prediction] += 1
			# # Batch updating 
			# tot_grads /= self.train_size
			# # Update historical gradient vector
			# adjusted_grads = tot_grads / (fudge_factor + np.sqrt(history_grads))
			# brnn.update_params(adjusted_grads, learn_rate)
			# End of the core AdaGrad updating algorithm
			accuracy = tot_count / float(self.train_size)
			logger.debug('Epoch %d, total cost: %f, overall accuracy: %f' % (i, tot_error, accuracy))
			logger.debug('Confusion matrix: ')
			logger.debug(conf_matrix)
			logger.debug('-' * 50)
			if (i+1) % 10 == 0:
				logger.debug('=' * 50)
				logger.debug('Test at epoch: %d' % i)
				# Testing
				tot_count = 0
				for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
					prediction = brnn.predict(test_seq)[0]
					tot_count += test_label == prediction
				logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
				logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
				logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
				logger.debug('=' * 50)
				BRNN.save('sentiment.nbrnn.Oct.pkl', brnn)
		end_time = time.time()
		logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
		# Testing
		tot_count = 0
		for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
			prediction = brnn.predict(test_seq)[0]
			tot_count += test_label == prediction
		logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
		logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
		logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
		# Re-testing on training set
		tot_count = 0
		for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
			prediction = brnn.predict(train_seq)[0]
			tot_count += train_label == prediction
		logger.debug('Training accuracy re-testing: %f' % (tot_count / float(self.train_size)))
		# Save BRNN
		BRNN.save('sentiment.nbrnn.Oct.pkl', brnn)
		logger.debug('Model successfully saved...')

	@unittest.skip('Wait a minute')
	def testTBRNNwithFineTuning(self):
		# Set print precision
		np.set_printoptions(threshold=np.nan)

		config_filename = './sentiment_brnn.conf'
		start_time = time.time()
		configer = RNNConfiger(config_filename)
		brnn = TBRNN(configer, verbose=True)
		# brnn = TBRNN.load('sentiment.brnn.Sep5.pkl')
		end_time = time.time()
		logger.debug('Time used to load TBRNN: %f seconds.' % (end_time-start_time))
		logger.debug('Start training TBRNN with fine-tuning...')
		# Training
		logger.debug('positive labels: %d' % np.sum(self.senti_train_label))
		logger.debug('negative labels: %d' % (self.senti_train_label.shape[0]-np.sum(self.senti_train_label)))
		start_time = time.time()
		## AdaGrad learning algorithm instead of the stochastic gradient descent algorithm
		# history_grads = np.zeros(brnn.num_params)
		n_epoch = 2000
		learn_rate = 1e-2
		embed_learn_rate = 1e-3
		fudge_factor = 1e-6
		for i in xrange(n_epoch):
			tot_count = 0
			tot_error = 0.0
			conf_matrix = np.zeros((2, 2), dtype=np.int32)
			tot_grads = np.zeros(brnn.num_params)
			logger.debug('Total number of parameters in TBRNN: %d' % brnn.num_params)
			for train_indices, train_label in zip(self.senti_train_words_label, self.senti_train_label):
				# Dynamically build training instances
				train_seq = self.word_embedding._embedding[train_indices, :]
				# Compute cost and gradients with respect to parameters and word-embeddings
				cost, current_grads = brnn.compute_cost_and_gradient(train_seq, [train_label])
				input_grads = brnn.compute_input_gradient(train_seq, [train_label])
				# Accumulating errors and gradients
				tot_grads += current_grads
				tot_error += cost
				# historical gradient accumulation
				# history_grads += current_grads ** 2
				# predict current training label
				prediction = brnn.predict(train_seq)[0]
				tot_count += prediction == train_label
				conf_matrix[train_label, prediction] += 1
				# Update word-embedding
				for k, j in enumerate(train_indices):
					self.word_embedding._embedding[j, :] -= embed_learn_rate * input_grads[k, :]
			# Batch updating 
			tot_grads /= self.train_size
			# Update historical gradient vector
			# adjusted_grads = tot_grads / (fudge_factor + np.sqrt(history_grads))
			# brnn.update_params(adjusted_grads, learn_rate)
			brnn.update_params(tot_grads, learn_rate)
			# End of the core AdaGrad updating algorithm
			accuracy = tot_count / float(self.train_size)
			logger.debug('Epoch %d, total cost: %f, overall accuracy: %f' % (i, tot_error, accuracy))
			logger.debug('Confusion matrix: ')
			logger.debug(conf_matrix)
			logger.debug('-' * 50)
			if (i+1) % 100 == 0:
				logger.debug('=' * 50)
				logger.debug('Test at epoch: %d' % i)
				# Testing
				tot_count = 0
				for test_indices, test_label in zip(self.senti_test_words_label, self.senti_test_label):
					# Dynamically build test instances
					test_seq = self.word_embedding._embedding[test_indices, :]
					prediction = brnn.predict(test_seq)[0]
					tot_count += test_label == prediction
				logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
				logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
				logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
				logger.debug('=' * 50)
		end_time = time.time()
		logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
		# Testing
		tot_count = 0
		for test_indices, test_label in zip(self.senti_test_set, self.senti_test_label):
			# Dynamically build test instances
			test_seq = self.word_embedding._embedding[test_indices, :]
			prediction = brnn.predict(test_seq)[0]
			tot_count += test_label == prediction
		logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
		logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.senti_test_label==1) / float(self.test_size)))
		logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.senti_test_label==0) / float(self.test_size)))
		# Re-testing on training set
		tot_count = 0
		for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
			prediction = brnn.predict(train_seq)[0]
			tot_count += train_label == prediction
		logger.debug('Training accuracy re-testing: %f' % (tot_count / float(self.train_size)))
		# Save new word-embedding on sentiment analysis task
		WordEmbedding.save('word-embedding-sentiment.pkl', )
		# Save TBRNN
		TBRNN.save('sentiment.brnn.finetune.Sep5_1.pkl', brnn)
		logger.debug('Model successfully saved...')

if __name__ == '__main__':
	unittest.main()
