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
		batch_size = 200
		image_row, image_col = 28, 28
		input = T.tensor4(name='input')
		input.reshape((batch_size, 1, image_row, image_col))
		truth = T.ivector()
		convNet = ConvNet(input, truth, self.configer, verbose=True)


if __name__ == '__main__':
	unittest.main()