#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-05 10:06:43
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import unittest
import theano
import theano.tensor as T
from pprint import pprint
sys.path.append('../source/')

from cnn import ConvNet
from config import CNNConfiger

class TestCNN(unittest.TestCase):
	def setUp(self):
		fname = '../mnist.conf'
		self.configer = CNNConfiger(fname)

	def testBuilding(self):
		convNet = ConvNet(self.configer, verbose=True)


if __name__ == '__main__':
	unittest.main()