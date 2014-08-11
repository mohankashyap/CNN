#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-09 09:15:23
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.1

import os, sys
sys.path.append('../source/')
import unittest
import cPickle
import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.io as sio
import numpy as np
	
from activations import Activation
from wordvec import WordEmbedding
from pprint import pprint
from utils import floatX
from mlp import AutoEncoder, DAE
from config import DAEConfiger

class TestSent(unittest.TestCase):
	def setUp(self):
		train_set_filename = '../data/sentiment_train.mat'
		test_set_filename = '../data/sentiment_test.mat'
		embedding_filename = '../data/wiki_embeddings.txt'
		word_embedding = WordEmbedding(embedding_filename)
		self.word_embedding = word_embedding
		self.senti_train_set = sio.loadmat(train_set_filename)['data'][0]
		self.senti_test_set = sio.loadmat(test_set_filename)['data'][0]
		pprint('%d sentences in the training set.' % len(self.senti_train_set))
		pprint('%d sentences in the test set.' % len(self.senti_test_set))

	@unittest.skip('Try to learn a non-linear mapping upon the auto-encoder, failed.')
	def testGenerate(self):
		'''
		Try using Denoising AutoEncoder to naivelly compressing and reconstructing 
		sentences.
		'''
		ae = AutoEncoder.load('./wordvec.ae')
		start_time = time.time()
		logfile = file('log.txt', 'wb')
		dict_size, embed_dim = self.word_embedding.dict_size(), self.word_embedding.embedding_dim()
		vectors = self.word_embedding.embedding
		recons_vectors = ae.reconstruct(vectors)
		pprint('Finished reconstructing, start to computing distances...')
		# Overall mappings
		mappings = dict()
		# Compute distance
		for i in xrange(dict_size):
			recons_vec = recons_vectors[i, :]
			diff = np.sum((vectors-recons_vec)**2, axis=1)
			m = np.argmin(diff)
			mappings[self.word_embedding.index2word[i]] = \
					self.word_embedding.index2word[m] 
			pprint('%s --> %s' % (self.word_embedding.index2word[i], 
								self.word_embedding.index2word[m]))
		logfile = file('./log.txt', 'wb')
		pprint(mappings, logfile)
		logfile.close()
		end_time = time.time()
		pprint('Time used to find mapping: %f seconds.' % (end_time-start_time))

	@unittest.skip('Build data finished...')
	def testBuild(self):
		'''
		Load training and test texts in sentiment analysis task, 
		preprocessing.
		'''
		np.random.seed(1991)
		senti_train_set_filename = '../data/sentiment_train_txt.txt'
		senti_test_set_filename = '../data/sentiment_test_txt.txt'
		embedding_filename = '../data/wiki_embeddings.txt'
		# Load training/test data sets and wiki-embeddings
		word_embedding = WordEmbedding(embedding_filename)
		self.word_embedding = word_embedding
		with file(senti_train_set_filename) as fin:
			senti_train_txt = fin.readlines()
		with file(senti_test_set_filename) as fin:
			senti_test_txt = fin.readlines()
		train_size = len(senti_train_txt)
		test_size = len(senti_test_txt)
		# Check size
		pprint('Training size: %d' % train_size)
		pprint('Test size: %d' % test_size)
		# Mapping word embedding
		self.senti_train_set, self.senti_test_set = [], []
		# Embedding for training set
		start_time = time.time()
		for i, sent in enumerate(senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_train_set.append(vectors)
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_test_set.append(vectors)
		end_time = time.time()
		pprint('Time used for mapping word embedding: %f minutes.' % ((end_time-start_time)/60))
		sio.savemat('../data/sentiment_train.mat', {'data' : self.senti_train_set})
		sio.savemat('../data/sentiment_test.mat', {'data' : self.senti_test_set})
		pprint('Data saved in matlab form...')

	@unittest.skip('AutoEncoder training finished...')
	def testTrain(self):
		# Setting parameters for AutoEncoder
		input_matrix = T.matrix(name='input')
		num_in, num_out = 50, 20
		act = Activation('sigmoid')
		is_denoising, is_sparse = True, False
		lambda1, mask = 1e-4, 0.7
		rng = RandomStreams(42)
		nepoch = 50
		learn_rate = 5
		# Build auto-encoder
		start_time = time.time()
		ae = AutoEncoder(input_matrix, (num_in, num_out), act, 
				is_denoising, is_sparse, lambda1, mask, rng, verbose=True)
		end_time = time.time()
		pprint('Time used to build AutoEncoder for word embedding: %f minutes.' %((end_time-start_time)/60))
		start_time = time.time()
		batch_size = 10000
		num_batches = self.word_embedding.dict_size() / batch_size
		for i in xrange(nepoch):
			rate = learn_rate
			tot_cost = 0.0
			for j in xrange(num_batches):
				train_set = self.word_embedding.embedding[j*batch_size : (j+1)*batch_size, :]
				cost = ae.train(train_set, rate)
				tot_cost += cost
			pprint('epoch %d, total cost = %f' % (i, tot_cost))
			AutoEncoder.save('./wordvec.ae', ae)
		end_time = time.time()
		pprint('Time used to train AutoEncoder: %f minutes.' % ((end_time-start_time)/60))


	def testAEwithZpadding(self):
		'''
		Here we try to attack the task of sentence generation by using AutoEncoder/Deep AutoEncoder and combined with supervised 
		linear mapping. The architecture suggests first using stacked auto-encoders to learn a hidden representation and then 
		try to recover the original sentence by a linear projection onto the original space.
		'''
		# Find the longest sentence in the training and test data set
		max_length = 0
		max_length = max(max_length, max([len(sentence) for sentence in self.senti_train_set]))
		max_length = max(max_length, max([len(sentence) for sentence in self.senti_test_set]))
		pprint('Longest sentence in both training and test part.')
		input_dim = self.word_embedding.embedding_dim() * max_length
		# Build training and test data set with 0 padding
		aug_senti_train_set = np.zeros((self.senti_train_set.shape[0], input_dim), dtype=floatX)
		aug_senti_test_set = np.zeros((self.senti_test_set.shape[0], input_dim), dtype=floatX)
		# Training set size and test set size
		start_time = time.time()
		for i, sent in enumerate(self.senti_train_set):
			length = np.prod(sent.shape)
			aug_senti_train_set[i, :length] = sent.reshape(1, -1)
		for i, sent in enumerate(self.senti_test_set):
			length = np.prod(sent.shape)
			aug_senti_test_set[i, :length] = sent.reshape(1, -1)
		end_time = time.time()
		pprint('Time used to build the vectorized array: %f seconds.' % (end_time-start_time))
		# Save padding matrices
		sio.savemat('senti_train_set.mat', {'data' : aug_senti_train_set)})
		sio.savemat('senti_test_set.mat', {'data' : aug_senti_test_set})
		pprint('Padded matrices saved...')
		# Train an auto-encoder with last layer as a linear mapping
		


if __name__ == '__main__':
	unittest.main()





