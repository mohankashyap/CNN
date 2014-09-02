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
from logistic import SoftmaxLayer

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
									  size=(fan_out, fan_out)), dtype=floatX),
					name='U', borrow=True)
		# Bias parameter for the hidden-layer encoder of RNN
		self.b = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='b', borrow=True)
		# h[0], zero vector
		self.h0 = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='h0', borrow=True)
		# Save all the parameters
		self.params = [self.W, self.U, self.b, self.h0]
		# recurrent function used to compress a sequence of input vectors
		# the first dimension should correspond to time
		def step(x_t, h_tm1):
			h_t = self.act.activate(T.dot(x_t, self.W) + \
									T.dot(h_tm1, self.U) + self.b)
			return h_t
		# h is the hidden representation over a time sequence
		self.hs, _ = theano.scan(fn=step, sequences=self.input, outputs_info=[self.h0])
		self.h = self.hs[-1]
		# L1, L2 regularization
		self.L1_norm = T.sum(T.abs_(self.W) + T.abs_(self.U))
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


class TBRNN(object):
	'''
	Bidirectional RNN with tied weights. This is just a trial for using 
	BRNN as a tool for sentence modeling.

	First trial on the task of sentiment analysis.
	'''
	def __init__(self, configs, verbose=True):
		if verbose: pprint('Build Tied weights Bidirectional Recurrent Neural Network')
		self.input = T.matrix(name='input')
		self.truth = T.ivector(name='label')
		self.learn_rate = T.scalar(name='learn rate')
		# Configure Activation function
		self.act = Activation(configs.activation)
		# Build bidirectional RNN with tied weights
		fan_in, fan_out = configs.num_input, configs.num_hidden
		# Tied weights:
		# 1, Feed-forward matrix: W
		# 2, Recurrent matrix: U
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
									  high=np.sqrt(6.0/(fan_in+fan_out)),
									  size=(fan_in, fan_out)), dtype=floatX),
					name='W', borrow=True)
		self.U = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
									  high=np.sqrt(6.0/(fan_in+fan_out)),
									  size=(fan_in, fan_out)), dtype=floatX),
					name='U', borrow=True)
		# Bias parameter for the hidden-layer encoder of RNN
		self.b = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='b', borrow=True)
		# h[0], zero vector, treated as constants
		self.h_start = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='h_start', borrow=True)
		self.h_end = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='h_end', borrow=True)
		# Save all the parameters
		self.params = [self.W, self.U, self.b]
		# recurrent function used to compress a sequence of input vectors
		# the first dimension should correspond to time
		def step(x_t, h_tm1):
			h_t = self.act.activate(T.dot(x_t, self.W) + \
									T.dot(h_tm1, self.U) + self.b)
			return h_t
		# Forward and backward representation over time
		self.forward_h, _ = theano.scan(fn=step, sequences=self.input, outputs_info=[self.h_start])
		self.backward_h, _ = theano.scan(fn=step, sequences=self.input, outputs_info=[self.h_end], go_backwards=True)
		# Store the final value
		self.h_start_star = self.forward_h[-1]
		self.h_end_star = self.backward_h[-1]
		# L1, L2 regularization
		self.L1_norm = T.sum(T.abs_(self.W) + T.abs_(self.U))
		self.L2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)
		# Build function to show the learned representation for different sentences
		self.show_forward = theano.function(inputs=[self.input], outputs=self.h_start_star)
		self.show_backward = theano.function(inputs=[self.input], outputs=self.h_end_star)
		##################################################################################
		# Correlated BRNN
		##################################################################################
		# Concatenate these two vectors into one
		self.h = T.concatenate([self.h_start_star, self.h_end_star], axis=0)
		# Use concatenated vector as input to the Softmax/MLP classifier
		self.softmax = SoftmaxLayer(self.h, (2*configs.num_hidden, configs.num_class))
		self.params.extend(self.softmax.params)
		# Build cost function
		self.cost = self.softmax.NLL_loss(self.truth)
		if configs.regularization:
			self.cost += configs.lambda1 * self.L2_norm
		# Compute gradient
		self.gradparams = T.grad(self.cost, self.params)
		self.updates = []
		for param, gradparam in zip(self.params, self.gradparams):
			self.updates.append((param, param-self.learn_rate*gradparam))
		# Build objective function
		self.objective = theano.function(inputs=[self.input, self.truth, self.learn_rate], \
										 outputs=self.cost, updates=self.updates)
		# Compute the gradients
		self.compute_gradient = theano.function(inputs=[self.input, self.truth], 
												outputs=self.gradparams)
		# Build prediction function
		self.predict = theano.function(inputs=[self.input], outputs=self.softmax.pred)
		if verbose:
			pprint('*' * 50)
			pprint('Finished constructing Tied weights Bidirectional Recurrent Neural Network (TBRNN)')
			pprint('Size of input dimension: %d' % configs.num_input)
			pprint('Size of hidden/recurrent dimension: %d' % configs.num_hidden)
			pprint('Size of output dimension: %d' % configs.num_class)
			pprint('Is regularization applied? %s' % ('yes' if configs.regularization else 'no'))
			if configs.regularization:
				pprint('Coefficient of regularization term: %f' % configs.lambda1)
			pprint('*' * 50)
		# Checking some important variables
		self.check_gradient = theano.function(inputs=[self.input, self.truth], 
											  outputs=self.gradparams)

	def train(self, input, truth, learn_rate):
		cost = self.objective(input, truth, learn_rate)
		return cost

	# This method is used to implement the batch updating algorithm
	def update_params(self, gradparams, learn_rate):
		for param, gradparam in zip(self.params, gradparams):
			# Updating using stochastic gradient descent
			param.set_value(param.get_value(borrow=True)-learn_rate*gradparam)

	@staticmethod
	def save(fname, model):
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		with file(fname, 'rb') as fin:
			return cPickle.load(fin)
