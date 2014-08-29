#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 17:54:41
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import unittest
import numpy as np
import time
import theano
import theano.tensor as T
from pprint import pprint
sys.path.append('../source/')

from rnn import RNN
from config import RNNConfiger
from utils import floatX

class TestRNN(unittest.TestCase):
	def setUp(self):
		# Load parameters of RNNConfiger and create RNN
		self.configer = RNNConfiger('testrnn.conf')
		pprint('Parameters loaded by RNNConfiger: ')
		pprint('=' * 50)
		pprint('Input dimension of RNN: %d' % self.configer.num_input)
		pprint('Hidden dimension of RNN: %d' % self.configer.num_hidden)
		pprint('Regularizer parameter for L1-norm: %f' % self.configer.lambda1)
		pprint('Regularizer parameter for L2-norm: %f' % self.configer.lambda2)
		# Construct RNN
		start_time = time.time()
		self.rnn = RNN(configs=self.configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build the architecture of RNN: %f seconds.' % (end_time-start_time))

	def testCompress(self):
		rand_input = np.random.rand(100, 50)
		rnn_output = self.rnn.compress(rand_input)
		# Compute compressed vector by numpy
		rnn_h0 = self.rnn.h0.get_value()		
		rnn_b = self.rnn.b.get_value()
		rnn_W = self.rnn.W.get_value()
		rnn_U = self.rnn.U.get_value()
		# manually check whether the function is correct or not
		tmp_hidden = rnn_h0
		tmp_hidden.shape = (1, -1)
		rnn_b.shape = (1, -1)
		seq_length = rand_input.shape[0]
		def sigmoid(x):
			return 1.0 / (1.0 + np.exp(-x))
		for i in xrange(seq_length):
			tmp_hidden = sigmoid(np.dot(rand_input[i, :], rnn_W) + \
											np.dot(tmp_hidden, rnn_U) + rnn_b)
		diff_L1 = np.sum(np.abs(tmp_hidden-rnn_output))
		diff_L2 = np.sum((tmp_hidden-rnn_output) ** 2)
		pprint('Difference in L1 norm: %f' % diff_L1)
		pprint('Difference in L2 norm: %f' % diff_L2)

if __name__ == '__main__':
	unittest.main()


