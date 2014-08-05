#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 17:54:38
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
sys.path.append('../source/')
import cPickle
import unittest
from pprint import pprint

class TestMNIST(unittest.TestCase):
	def setUp(self):
		fname = '../data/mnist.pkl'
		fin = file(fname, 'rb')
		self.train_set, self.valid_set, self.test_set = cPickle.load(fin)
		fin.close()

	def testData(self):
		pprint("Size of Training set: %d" % len(self.train_set[0]))
		pprint("Size of Validation set: %d" % len(self.valid_set[0]))
		pprint("Size of Test set: %d" % len(self.test_set[0]))


if __name__ == '__main__':
	unittest.main()