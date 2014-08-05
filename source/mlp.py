#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 14:51:23
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import utils
import numpy as np
import theano
import theano.tensor as T
from utils import floatX

class HiddenLayer(object):
	'''
	Multilayer Perceptron
	'''
	def __init__(self, input, (num_in, num_out), act=None):
		'''
		@input: theano symbolic tensor. Input to hidden layer, a 2nd order tensor, i.e., 
				a matrix of size (num_in, num_example), where each column is an instance.
		@num_in: Int. Size of input dimension.
		@num_out: Int. Size of output dimension.
		@act: Activation. Activation function used at each neuron.
		'''
		self.input = input
		fan_in = num_in
		fan_out = num_out
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
									  high=np.sqrt(6.0/(fan_in+fan_out)),
									  size=(num_out, num_in)), dtype=floatX),
					name='W', borrow=True)
		self.b = theano.shared(value=np.zeros(num_out, dtype=floatX), name='b', borrow=True)
		self.output = act.activate(T.dot(self.W, self.input) + self.b)
		# Stack parameters
		self.param = [self.W, self.b]

	def L2_loss(self):
		return T.sum(self.W ** 2)