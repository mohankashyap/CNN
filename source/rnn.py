#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 10:37:00
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import numpy as np
import theano
import theano.tensor as T
import time
import cPickle
from pprint import pprint

import config
import utils
from utils import floatX
from activations import Activation

class RNN(object):
	'''
	Basic component for Recurrent Neural Network
	'''
	def __init__(self, configs=None, verbose=True):
		'''
		Basic RNN is an unsupervised component, where the input is a sequence and the 
		output is a vector with fixed length
		'''
		if verbose: pprint('Build Recurrent Neural Network...')
		self.input = T.matrix(name='input', dtype=floatX)
		self.learn_rate = T.scalar(name='learn rate')		
		# Configure activation function
		self.act = Activation(configs.activation)
		fan_in = configs.num_input
		fan_out = configs.num_hidden
		# Initialize all the variables in RNN, including:
		# 1, Feed-forward matrix, feed-forward bias, W, W_b
		# 2, Recurrent matrix, recurrent bias, U, U_b
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
									  high=np.sqrt(6.0/(fan_in+fan_out)), 
									  size=(fan_in, fan_out)), dtype=floatX),
					name='W', borrow=True)
		self.U = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_out+fan_out)),
									  high=np.sqrt(6.0/(fan_out+fan_out)),
									  size=(fan_out, fan_out)), dtype=floatX,
					name='U', borrow=True)
		self.b = theano.shared(value=np.zeros(fan_out), dtype=floatX, name='W_b', borrow=True)
		# h[0], zero vector
		self.h0 = theano.shared(value=np.zeros(fan_out), dtype=floatX, name='h0', borrow=True)
		# Save all the parameters
		self.params = [self.W, self.U, self.b, self.h0]
		# recurrent function used to compress a sequence of input vectors
		# the first dimension should correspond to time
		def step(x_t, h_tm1):
			h_t = self.act.activate(T.dot(x_t, self.W) + \
									T.dot(h_tm1, self.U) + self.b)
			return h_t
		# h is the hidden representation over a time sequence
		self.h, _ = theano.scan(fn=step, sequences=self.input, outputs_info=[self.h0])
		# L1, L2 regularization
		self.L1_norm = T.sum(T.abs(self.W) + T.abs(self.U))
		self.L2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)
		# Compress function
		self.compress = theano.function(inputs=[self.input], outputs=self.h)

	@staticmethod
	def save(fname, model):
		'''
		Save current RNN model into fname
		@fname: String. Filename to save the model.
		@model: RNN. An instance of RNN class.
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		Load an RNN model from fname
		@fname: String. Filename to load the model.
		'''
		with file(fname, 'rb') as fin:
			return cPickle.load(fin)




