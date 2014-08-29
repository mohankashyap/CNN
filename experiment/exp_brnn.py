#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 19:06:51
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import csv
import numpy as np
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
		np.random.seed(1991)
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
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_train_len.append(len(words))
			self.senti_train_set.append(vectors)
		# Embedding for test set
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			senti_test_len.append(len(words))
			self.senti_test_set.append(vectors)
		assert senti_train_len == [seq.shape[0] for seq in self.senti_train_set]
		assert senti_test_len == [seq.shape[0] for seq in self.senti_test_set]
		end_time = time.time()
		pprint('Time used to build initial training and test matrix: %f seconds.' % (end_time-start_time))
		# Store data
		self.train_size = train_size
		self.test_size = test_size
		self.word_embedding = word_embedding

	def testBRNNonSentiment(self):
		config_filename = './sentiment_brnn.conf'
		start_time = time.time()
		configer = RNNConfiger(config_filename)
		brnn = TBRNN(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build TBRNN: %f seconds.' % (end_time-start_time))
		n_epoch = 1000
		learn_rate = 0.1
		# Training
		pprint('positive labels: %d' % np.sum(self.senti_train_label))
		pprint('negative labels: %d' % (self.senti_train_label.shape[0]-np.sum(self.senti_train_label)))
		start_time = time.time()
		for i in xrange(n_epoch):
			tot_count = 0
			tot_error = 0.0			
			conf_matrix = np.zeros((2, 2), dtype=np.int32)
			for train_seq, train_label in zip(self.senti_train_set, self.senti_train_label):
				cost, accuracy = brnn.train(train_seq, [train_label], learn_rate)
				tot_count += accuracy
				tot_error += cost
				conf_matrix[train_label, 1-int(accuracy)] += 1
			accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / float(np.sum(conf_matrix))
			pprint('Epoch %d, total cost: %f, overall accuracy: %f' % (i, tot_error, accuracy))
			pprint('Confusion matrix: ')
			pprint(conf_matrix)
			pprint('-' * 50)
		end_time = time.time()
		pprint('Time used for training: %f minutes.' % ((end_time-start_time)/60))
		# Testing
		tot_count = 0
		for test_seq, test_label in zip(self.senti_test_set, self.senti_test_label):
			prediction = brnn.predict(test_seq)
			tot_count += test_label == prediction
		pprint('Test accuracy: %f' % (tot_count / float(self.test_size))) 



if __name__ == '__main__':
	unittest.main()
