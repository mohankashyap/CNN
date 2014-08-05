#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-05 08:00:32
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import unittest
from pprint import pprint
sys.path.append('../source/')

from config import CNNConfiger


class TestCNNConfiger(unittest.TestCase):
	def setUp(self):
		fname = '../mnist.conf'
		self.configer = CNNConfiger(fname)

	def testValue(self):
		self.assertEqual(self.configer.activation, 'sigmoid', 
						'Activation function load error')
		self.assertEqual(self.configer.num_convpool, 2, 
						'Number of convolution and pooling layers is wrong')
		self.assertEqual(self.configer.num_mlp, 1, 
						'Number of multilayer perceptron is wrong')
		self.assertEqual(self.configer.num_softmax, 1, 
						'Number of softmax layer is wrong')
		self.assertAlmostEqual(self.configer.learning_rate, 0.1, 
						'Learning rate is wrong')
		self.assertEqual(self.configer.batch_size, 200, 
						'Batch size is wrong')
		pprint('Architecture for Convolution and Pooling layers: ')
		pprint(self.configer.convs)
		pprint(self.configer.pools)
		pprint('Architecture for Multilayer perceptron layers: ')
		pprint(self.configer.mlps)
		pprint('Architecture for Softmax layers: ')
		pprint(self.configer.softmaxs)


if __name__ == '__main__':
	unittest.main()