#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 17:54:38
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
sys.path.add('../source/')
import cPickle
import unittest
from pprint import pprint

class TestMNIST(object):
	def setUp(self):
		fname = '../data/mnist.pkl'
		self.train_set, self.valid_set, self.test_set = cPickle.load(fname)
		pprint("Size of Training set: %d" % len(self.train_set))
		pprint("Size of Validation set: %d" % len(self.valid_set))
		pprint("Size of Test set: ")