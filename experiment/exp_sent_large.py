#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-18 08:54:09
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import gzip
sys.path.append('../source/')
sys.path.append('../data/')
import unittest
import cPickle
import time
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import scipy.io as sio
import numpy as np

from logistic import SoftmaxLayer
from activations import Activation
from wordvec import WordEmbedding
from pprint import pprint
from utils import floatX
from mlp import AutoEncoder, DAE, HiddenLayer
from config import DAEConfiger
from exp_sentence import SentModel


class TestOnLarge(unittest.TestCase):
	'''
	Test the initial sentence model on Wiki-Data set, which contains 
	39,746,550 sentences,
	782,603,381 words
	'''
	def setUp(self):
		train_txt_filename = '../data/wiki_sentence.txt'
		wiki_filename = '../data/wiki_embeddings.txt'
		start_time = time.time()
		self.word_embedding = WordEmbedding(wiki_filename)
		with file(train_txt_filename, 'rb') as fin:
			self.train_txt = fin.readlines()
		end_time = time.time()
		# Since the maximum length in the task of sentiment_analysis is 56, during training
		# we will set 56 as the maximum length of each sentence
		self.max_length = 56
		self.num_sent = len(self.train_txt)
		self.batch_size = 2000
		self.nepoch = 5
		pprint('Time used to load wiki sentences into memory: %f seconds.' % (end_time-start_time))
		pprint('Number of sentences in the data set: %d' % len(self.train_txt))

	def testTrain(self):
		'''
		Train Auto-Encoder + SoftmaxLayer on batch learning mode.
		'''
		input_dim, hidden_dim = self.max_length * self.word_embedding.embedding_dim(), 500
		# Build AutoEncoder + SoftmaxLayer
		start_time = time.time()
		seed = 1991
		input_matrix = T.matrix(name='input')
		num_in, num_out = input_dim, hidden_dim
		act = Activation('tanh')
		is_denoising, is_sparse = True, False
		lambda1, mask = 1e-4, 0.5
		rng = RandomStreams(seed)
		sent_model = SentModel(input_matrix, (num_in, num_out), act, 
				is_denoising, is_sparse, lambda1, mask, rng, verbose=True)
		end_time = time.time()
		pprint('Time used to build the model: %f seconds.' % (end_time-start_time))
		# Loading training data and start batch training mode
		num_batch = self.num_sent / self.batch_size
		learn_rate = 0.1
		# Pretraining
		pprint('Start pretraining...')
		start_time = time.time()
		for i in xrange(self.nepoch):
			# Batch training
			pprint('Training epoch: %d' % i)
			for j in xrange(num_batch):
				train_set = np.zeros((self.batch_size, self.max_length * self.word_embedding.embedding_dim()), dtype=floatX)
				train_txt = self.train_txt[j*self.batch_size : (j+1)*self.batch_size]
				for k, sent in enumerate(train_txt):
					words = sent.split()
					vectors = np.asarray([self.word_embedding.wordvec(word) for word in words])
					vectors = vectors.flatten()
					train_set[k, :vectors.shape[0]] = vectors
				rate = learn_rate
				cost = sent_model.pretrain(train_set, rate)
				if (j+1) % 500 == 0:
					pprint('Training epoch: %d, Number batch: %d, cost = %f' % (i, j, cost))
			# Saving temporary pretraining model in .gz
			with gzip.GzipFile('./large_pretrain.sent.gz', 'wb') as fout:
				cPickle.dump(sent_model, fout)
		end_time = time.time()
		pprint('Time used for pretraining: %f minutes.' % ((end_time-start_time)/60.0))
		# Fine tuning
		pprint('Start fine-tuning...')
		start_time = time.time()
		for i in xrange(self.nepoch):
			# Batch training
			pprint('Training epoch: %d' % i)
			for j in xrange(num_batch):
				train_set = np.zeros((self.batch_size, self.max_length * self.word_embedding.embedding_dim()), dtype=floatX)
				train_txt = self.train_txt[j*self.batch_size : (j+1)*self.batch_size]
				for k, sent in enumerate(train_txt):
					words = sent.split()
					vectors = np.asarray([self.word_embedding.wordvec(word) for word in words])
					vectors = vectors.flatten()
					train_set[k, :vectors.shape[0]] = vectors
				rate = learn_rate
				cost = sent_model.finetune(train_set, rate)
				if (j+1) % 500 == 0:
					pprint('Training epoch: %d, Number batch: %d, cost = %f' % (i, j, cost))
			# Saving temporary fine-tuning model in .gz
			with gzip.GzipFile('./large_finetune.sent.gz', 'wb') as fout:
				cPickle.dump(sent_model, fout)
		end_time = time.time()
		pprint('Time used for fine-tuning: %f minutes.' %((end_time-start_time)/60.0))


if __name__ == '__main__':
	unittest.main()


