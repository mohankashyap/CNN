#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 10:14:43
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Version : 0.0

import os, sys
import theano
import utils
import numpy as np
from utils import floatX

class WordEmbedding(object):
	def __init__(self, fname):
		'''
		@fname:	String. File path to the wiki-word embedding file
		'''
		with file(fname, 'r') as fin:
			line = fin.readline()
			self._dict_size, self._embed_dim = [int(s) for s in line.split()]
			self._embedding = np.zeros((self._dict_size, self._embed_dim), dtype=floatX) 		
			self._word2index = dict()
			self._index2word = []
			for i in xrange(self._dict_size):
				line = fin.readline().split()
				self._word2index[line[0]] = i
				self._index2word.append(line[0])
				self._embedding[i, :] = np.array([float(x) for x in line[1:]])

	# Getters
	def dict_size(self):
		return self._dict_size

	def embedding_dim(self):
		return self._embed_dim
	
	def words(self):
		return self._word2index.keys()

	@property
	def embedding(self):
	    return self._embedding
	
	def word2index(self, word):
		'''
		@word:	String. Return word index if word exists in the dictionary else -1
		'''
		idx = -1
		try:
			idx = self._word2index[word]
		except KeyError:
			pass
		return idx

	def index2word(self, index):
		'''
		@index: int. Return the corresponding word associated with index.
		'''
		assert 0 <= index < self._dict_size
		return self._index2word[index]

	def wordvec(self, word):
		'''
		@word: 	String. Return word vector of word if word exists in the dictonary 
				else return the word embedding of "unknown"
		'''
		idx = self._word2index['unknown']
		try:
			idx = self._word2index[word]
		except KeyError:
			pass
		return self._embedding[idx]
