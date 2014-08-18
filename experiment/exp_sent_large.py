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
		pprint('Time used to load wiki sentences into memory: %f seconds.' % (end_time-start_time))
		pprint('Number of sentences in the data set: %d' % len(self.train_txt))

	def testBuild(self):
		'''
		Build data matrix from WordEmbedding and wiki_sentence data file.
		'''
		train_data = np.zeros((self.num_sent, self.max_length), dtype=floatX)
		start_time = time.time()
		for i, sent in enumerate(self.train_txt):
			if (i+1) % 500 == 0:
				pprint('%d sentences have been processed...' % (i+1))
			words = sent.split()
			vectors = np.asarray([self.word_embedding.wordvec(word) for word in words])
			vectors = vectors.flatten()
			train_data[i, :vectors.shape[0]] = vectors
		end_time = time.time()
		sio.savemat('wiki_sentence.mat', {'data' : train_data})
		pprint('Time used to store the data matrix: %f minutes.' % ((end_time-start_time)/60.0))




if __name__ == '__main__':
	unittest.main()


