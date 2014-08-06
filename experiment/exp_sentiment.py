#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-06 13:49:13
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import cPickle
import unittest
sys.path.append('../source/')
from pprint import pprint

from utils import floatX
from cnn import ConvNet
from logistic import SoftmaxLayer
from mlp import MLP
from wordvec import WordEmbedding

class TestSentiment(unittest.TestCase):
	def setUp(self):
		'''
		Load training and test texts and labels 
		in sentiment analysis task, preprocessing.
		'''
		np.random.seed(1991)
		senti_train_set_filename = '../data/sentiment_train_txt.txt'
		senti_train_label_filename = '../data/sentiment_train_label.txt'
		senti_test_set_filename = '../data/sentiment_test_txt.txt'
		senti_test_label_filename = '../data/sentiment_test_label.txt'
		embedding_filename = '../data/wiki_embeddings.txt'
		# Load training/test data sets and wiki-embeddings
		word_embedding = WordEmbedding(embedding_filename)
		with file(senti_train_set_filename) as fin:
			senti_train_txt = fin.readlines()
		with file(senti_test_set_filename) as fin:
			senti_test_txt = fin.readlines()
		self.senti_train_label = np.loadtxt(senti_train_label_filename, dtype=np.int32)
		self.senti_test_label = np.loadtxt(senti_test_label_filename, dtype=np.int32)
		train_size = len(senti_train_txt)
		test_size = len(senti_test_txt)
		# Check size
		assert train_size == self.senti_train_label.shape[0]
		assert test_size == self.senti_test_label.shape[0]
		pprint('Training size: %d' % train_size)
		pprint('Test size: %d' % test_size)
		# Compute word embedding
		self.senti_train_set = np.zeros((train_size, word_embedding.embedding_dim()), dtype=floatX)
		self.senti_test_set = np.zeros((test_size, word_embedding.embedding_dim()), dtype=floatX)
		# Embedding for training set
		for i, sent in enumerate(senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			pprint('Trainging set, Number of words in sentence %d: %d' % (i, len(words)))
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_train_set[i, :] = np.mean(vectors, axis=0)
		# Embedding for test set
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			pprint('Test set, Number of words in sentence %d: %d' % (i, len(words)))
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_test_set[i, :] = np.mean(vectors, axis=0)
		# Shuffle training and test data set
		train_rand_index = np.random.permutation(train_size)
		test_rand_index = np.random.permutation(test_size)
		self.senti_train_set = self.senti_train_set[train_rand_index, :]
		self.senti_test_set = self.senti_test_set[test_rand_index, :]
		self.senti_train_label = self.senti_train_label[train_rand_index]
		self.senti_test_label = self.senti_test_label[test_rand_index]

	def testSoftmax(self):
		pass

	def testMLP(self):
		pass

	def testCNN(self):
		pass

	











