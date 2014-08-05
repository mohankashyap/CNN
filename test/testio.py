#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 10:46:36
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os, sys
import unittest
import numpy as np
from pprint import pprint
sys.path.append('../source/')

import utils
from wordvec import WordEmbedding

class TestIO(unittest.TestCase):
	def setUp(self):
		embedding_fname = '../data/wiki_embeddings.txt'
		snip_train_txt = '../data/train_snip.txt'
		snip_test_txt = '../data/test_snip.txt'
		snip_train_label = '../data/train_label.txt'
		snip_test_label = '../data/test_label.txt'
		self.word_embedding = WordEmbedding(embedding_fname)
		self.train_snip_txt = utils.loadtxt(snip_train_txt)
		self.train_snip_label = utils.loadlabel(snip_train_label)
		self.test_snip_txt = utils.loadtxt(snip_test_txt)
		self.test_snip_label = utils.loadlabel(snip_test_label)

	def testEmbedding(self):
		pprint('Size of word vocabulary: %d' % self.word_embedding.dict_size())
		pprint('Dimension of word embedding: %d' % self.word_embedding.embedding_dim())
		self.assertEqual(self.word_embedding.dict_size(), 311467, 
						'Incorrect size of word vocabulary')
		self.assertEqual(self.word_embedding.embedding_dim(), 50, 
						'Incorrect dimension of word embedding')
		pprint("Unknown: ")
		pprint(self.word_embedding.wordvec('unknown'))

	def testSnippetTrain(self):
		self.assertEqual(len(self.train_snip_txt), 10060, 
						'Training data not complete')
		self.assertEqual(len(self.train_snip_label), 10060, 
						'Training label not complete')
		num_class = len(set(self.train_snip_label))
		self.assertEqual(num_class, 8, 'Number of classes should be 8')
		for i in xrange(num_class):
			cls_count = np.sum((i+1) == self.train_snip_label)
			pprint("Number of instances in class %d: %d" % (i+1, cls_count))

	def testSnippetTest(self):
		self.assertEqual(len(self.test_snip_txt), 2280, 
						'Test data not complete')
		self.assertEqual(len(self.test_snip_label), 2280, 
						'Test label not complete')
		num_class = len(set(self.test_snip_label))
		self.assertEqual(num_class, 8, 'Number of classes should be 8')
		for i in xrange(num_class):
			cls_count = np.sum((i+1) == self.test_snip_label)
			pprint("Number of instances in class %d: %d" % (i+1, cls_count))

if __name__ == '__main__':
	unittest.main()

