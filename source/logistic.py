#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 15:09:16
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import utils
import numpy as np
import theano
import theano.tensor as T
from utils import floatX

class LogisticLayer(object):
	'''
	Logistic Regression Layer which performs two-class classification
	'''
	def __init__(self, input, num_in):
		'''
		@input: theano symbolic tensor. Input to the Logistic Regression layer, a
				vector of size (num_example, num_in), where each row of the 
				matrix is an input instance.
		@num_in: Int. Size of the input dimension. 
		'''
		self.input = input
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(num_in+1)),
									  high=np.sqrt(6.0/(num_in+1)),
									  size=(num_in)), dtype=floatX),
					name='W_logistic', borrow=True)
		self.b = theano.shared(value=np.zeros(1, dtype=floatX), name='b_logistic', borrow=True)
		self.output = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
		# Stack parameters
		self.params = [self.W, self.b]
		# Prediction for classification
		self.pred = self.output >= 0.5
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)

	def NLL_loss(self, truth):
		'''
		@truth: np.array. Truth label for each input instance.
		'''
		return -T.mean(truth * T.log(self.output) + (1-truth) * T.log(1-self.output))

	def L1_loss(self):
		return T.sum(T.abs_(self.W))

	def L2_loss(self):
		return T.sum(self.W ** 2)


class SoftmaxLayer(object):
	'''
	Softmax Layer which performs multi-class Logistic Regression
	'''
	def __init__(self, input, (num_in, num_out)):
		'''
		@input: theano symbolic tensor. Input to the Softmax layer, a 
				matrix of size (num_example, num_in), where each row of 
				the matrix is an input instance.
		@num_in: Int. Size of the input dimension.
		@num_out: Int. Size of the output dimension.
		'''
		self.input = input
		# Initialize 
		fan_in = num_in
		fan_out = num_out
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
									  high=np.sqrt(6.0/(fan_in+fan_out)),
									  size=(num_in, num_out)), dtype=floatX),
					name='W_softmax', borrow=True)
		self.b = theano.shared(value=np.zeros(num_out, dtype=floatX), name='b_softmax', borrow=True)
		self.output = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
		# Stack parameters
		self.params = [self.W, self.b]
		# Prediction for classification
		self.pred = T.argmax(self.output, axis=1)
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)

	def NLL_loss(self, ground_truth):
		'''
		@ground_truth: np.array. Truth label for each input instance.
		'''
		return -T.mean(T.log(self.output)[T.arange(ground_truth.shape[0]), ground_truth])

	def L1_loss(self):
		return T.sum(T.abs_(self.W))

	def L2_loss(self):
		return T.sum(self.W ** 2)

	def prediction(self):
		return self.pred


