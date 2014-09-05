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
		num_input, num_hidden, num_class = configs.num_input, configs.num_hidden, configs.num_class
		# Stack all the variables together into a vector in order to apply the batch updating algorithm
		num_params = num_input * num_hidden + \
					 num_hidden * num_hidden + \
					 num_hidden + \
					 2 * num_hidden * num_class + \
					 num_class
		self.num_params = num_params
		self.theta = theano.shared(value=np.zeros(num_params, dtype=floatX), name='theta', borrow=True)
		# Incremental index
		param_idx = 0
		# Tied weights:
		# 1, Feed-forward matrix: W
		self.W = self.theta[param_idx: param_idx+num_input*num_hidden].reshape((num_input, num_hidden))
		self.W.name = 'W_RNN'
		W_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_input+num_hidden)),
									  		  high=np.sqrt(6.0/(num_input+num_hidden)),
									  		  size=(num_input, num_hidden)), dtype=floatX)
		param_idx += num_input * num_hidden
		# 2, Recurrent matrix: U
		self.U = self.theta[param_idx: param_idx+num_hidden*num_hidden].reshape((num_hidden, num_hidden))
		self.U.name = 'U_RNN'
		U_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_input+num_hidden)),
									  		  high=np.sqrt(6.0/(num_input+num_hidden)),
									  		  size=(num_input, num_hidden)), dtype=floatX)
		param_idx += num_hidden * num_hidden
		# Bias parameter for the hidden-layer encoder of RNN
		self.b = self.theta[param_idx: param_idx+num_hidden]
		self.b.name = 'b_RNN'
		b_init = np.zeros(num_hidden, dtype=floatX)		
		param_idx += num_hidden
		# Weight matrix for softmax function
		self.W_softmax = self.theta[param_idx: param_idx+2*num_hidden*num_class].reshape((2*num_hidden, num_class))
		self.W_softmax.name = 'W_softmax'
		W_softmax_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(2*num_hidden+num_class)), 
													  high=np.sqrt(6.0/(2*num_hidden+num_class)),
													  size=(2*num_hidden, num_class)), dtype=floatX)
		param_idx += 2*num_hidden*num_class
		# Bias vector for softmax function
		self.b_softmax = self.theta[param_idx: param_idx+num_class]
		self.b_softmax.name = 'b_softmax'
		b_softmax_init = np.zeros(num_class, dtype=floatX)
		param_idx += num_class
		# Set all the default parameters into theta
		self.theta.set_value(np.concatenate([x.ravel() for x in 
			(W_init, U_init, b_init, W_softmax_init, b_softmax_init)]))
		assert param_idx == num_params
		# h[0], zero vector, treated as constants
		self.h_start = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h_start', borrow=True)
		self.h_end = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h_end', borrow=True)
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
		self.L1_norm = T.sum(T.abs_(self.W) + T.abs_(self.U) + T.abs_(self.W_softmax))
		self.L2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2) + T.sum(self.W_softmax ** 2)
		# Build function to show the learned representation for different sentences
		self.show_forward = theano.function(inputs=[self.input], outputs=self.h_start_star)
		self.show_backward = theano.function(inputs=[self.input], outputs=self.h_end_star)
		##################################################################################
		# Correlated BRNN
		##################################################################################
		# Concatenate these two vectors into one
		self.h = T.concatenate([self.h_start_star, self.h_end_star], axis=0)
		# Use concatenated vector as input to the Softmax/MLP classifier
		self.output = T.nnet.softmax(T.dot(self.h, self.W_softmax) + self.b_softmax)		
		self.pred = T.argmax(self.output, axis=1)
		# Build cost function
		self.cost = -T.mean(T.log(self.output)[T.arange(self.truth.shape[0]), self.truth])
		if configs.regularization:
			self.cost += configs.lambda1 * self.L2_norm
		# Compute gradient
		self.gradtheta = T.grad(self.cost, self.theta)
		# Build objective function
		# Compute the gradients
		self.compute_cost_and_gradient = theano.function(inputs=[self.input, self.truth], 
												outputs=[self.cost, self.gradtheta])
		# Build prediction function
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)
		if verbose:
			pprint('*' * 50)
			pprint('Finished constructing Tied weights Bidirectional Recurrent Neural Network (TBRNN)')
			pprint('Size of input dimension: %d' % configs.num_input)
			pprint('Size of hidden/recurrent dimension: %d' % configs.num_hidden)
			pprint('Size of output dimension: %d' % configs.num_class)
			pprint('Is regularization applied? %s' % ('yes' if configs.regularization else 'no'))
			if configs.regularization:
				pprint('Coefficient of regularization term: %f' % configs.lambda1)
			pprint('Number of free parameters in TBRNN: %d' % self.num_params)
			pprint('*' * 50)

	def train(self, input, truth, learn_rate):
		cost = self.objective(input, truth, learn_rate)
		return cost

	# This method is used to implement the batch updating algorithm
	def update_params(self, gradtheta, learn_rate):
		# gradparams is a single long vector which can be used to update self.theta
		# Learning algorithm: simple stochastic gradient descent
		theta = self.theta.get_value(borrow=True)
		self.theta.set_value(theta - learn_rate * gradtheta, borrow=True)

	@staticmethod
	def save(fname, model):
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		with file(fname, 'rb') as fin:
			return cPickle.load(fin)


class BRNN(object):
	'''
	Bidirectional RNN. This is just a trial for using 
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
		num_input, num_hidden, num_class = configs.num_input, configs.num_hidden, configs.num_class
		# Stack all the variables together into a vector in order to apply the batch updating algorithm
		# Since there are two directions for the RNN, all the weight matrix associated with RNN will be 
		# duplicated
		num_params = 2 * (num_input * num_hidden + \
					 num_hidden * num_hidden + \
					 num_hidden) + \
					 2 * num_hidden * num_class + \
					 num_class
		self.num_params = num_params
		self.theta = theano.shared(value=np.zeros(num_params, dtype=floatX), name='theta', borrow=True)
		# Incremental index
		param_idx = 0
		# 1, Feed-forward matrix for forward direction: W_forward
		self.W_forward = self.theta[param_idx: param_idx+num_input*num_hidden].reshape((num_input, num_hidden))
		self.W_forward.name = 'W_forward_RNN'
		W_forward_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_input+num_hidden)),
									  		  		  high=np.sqrt(6.0/(num_input+num_hidden)),
									  		  		  size=(num_input, num_hidden)), dtype=floatX)
		param_idx += num_input * num_hidden
		# 1, Feed-forward matrix for backward direction: W_backward
		self.W_backward = self.theta[param_idx: param_idx+num_input*num_hidden].reshape((num_input, num_hidden))
		self.W_backward.name = 'W_backward_RNN'
		W_backward_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_input+num_hidden)),
													   high=np.sqrt(6.0/(num_input+num_hidden)),
													   size=(num_input, num_hidden)), dtype=floatX)
		param_idx += num_input * num_hidden
		# 2, Recurrent matrix for forward direction: U_forward
		self.U_forward = self.theta[param_idx: param_idx+num_hidden*num_hidden].reshape((num_hidden, num_hidden))
		self.U_forward.name = 'U_forward_RNN'
		U_forward_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_hidden+num_hidden)),
													  high=np.sqrt(6.0/(num_hidden+num_hidden)),
													  size=(num_hidden, num_hidden)), dtype=floatX)
		param_idx += num_hidden * num_hidden
		# 2, Recurrent matrix for backward direction: U_backward
		self.U_backward = self.theta[param_idx: param_idx+num_hidden*num_hidden].reshape((num_hidden, num_hidden))
		self.U_backward.name = 'U_backward_RNN'
		U_backward_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(num_hidden+num_hidden)),
													   high=np.sqrt(6.0/(num_hidden+num_hidden)),
													   size=(num_hidden, num_hidden)), dtype=floatX)
		param_idx += num_hidden * num_hidden
		# 3, Bias parameter for the hidden-layer forward direction RNN
		self.b_forward = self.theta[param_idx: param_idx+num_hidden]
		self.b_forward.name = 'b_forward_RNN'
		b_forward_init = np.zeros(num_hidden, dtype=floatX)		
		param_idx += num_hidden
		# 3, Bias parameter for the hidden-layer backward direction RNN
		self.b_backward = self.theta[param_idx: param_idx+num_hidden]
		self.b_backward.name = 'b_backward_RNN'
		b_backward_init = np.zeros(num_hidden, dtype=floatX)
		param_idx += num_hidden
		# Weight matrix for softmax function
		self.W_softmax = self.theta[param_idx: param_idx+2*num_hidden*num_class].reshape((2*num_hidden, num_class))
		self.W_softmax.name = 'W_softmax'
		W_softmax_init = np.asarray(np.random.uniform(low=-np.sqrt(6.0/(2*num_hidden+num_class)), 
													  high=np.sqrt(6.0/(2*num_hidden+num_class)),
													  size=(2*num_hidden, num_class)), dtype=floatX)
		param_idx += 2*num_hidden*num_class
		# Bias vector for softmax function
		self.b_softmax = self.theta[param_idx: param_idx+num_class]
		self.b_softmax.name = 'b_softmax'
		b_softmax_init = np.zeros(num_class, dtype=floatX)
		param_idx += num_class
		# Set all the default parameters into theta
		self.theta.set_value(np.concatenate([x.ravel() for x in 
			(W_forward_init, W_backward_init, U_forward_init, U_backward_init, 
			 b_forward_init, b_backward_init, W_softmax_init, b_softmax_init)]))
		assert param_idx == num_params
		# h[0], zero vector, treated as constants
		self.h_start = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h_start', borrow=True)
		self.h_end = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h_end', borrow=True)
		# recurrent function used to compress a sequence of input vectors
		# the first dimension should correspond to time
		def forward_step(x_t, h_tm1):
			h_t = self.act.activate(T.dot(x_t, self.W_forward) + \
									T.dot(h_tm1, self.U_forward) + self.b_forward)
			return h_t
		def backward_step(x_t, h_tm1):
			h_t = self.act.activate(T.dot(x_t, self.W_backward) + \
									T.dot(h_tm1, self.U_backward) + self.b_backward)
			return h_t
		# Forward and backward representation over time
		self.forward_h, _ = theano.scan(fn=forward_step, sequences=self.input, outputs_info=[self.h_start])
		self.backward_h, _ = theano.scan(fn=backward_step, sequences=self.input, outputs_info=[self.h_end], go_backwards=True)
		# Store the final value
		self.h_start_star = self.forward_h[-1]
		self.h_end_star = self.backward_h[-1]
		# L1, L2 regularization
		self.L1_norm = T.sum(T.abs_(self.W_forward) + T.abs_(self.W_backward) + \
							 T.abs_(self.U_forward) + T.abs_(self.U_backward) + \
							 T.abs_(self.W_softmax))
		self.L2_norm = T.sum(self.W_forward ** 2) + T.sum(self.W_backward ** 2) + \
					   T.sum(self.U_forward ** 2) + T.sum(self.U_backward ** 2) + \
					   T.sum(self.W_softmax ** 2)
		# Build function to show the learned representation for different sentences
		self.show_forward = theano.function(inputs=[self.input], outputs=self.h_start_star)
		self.show_backward = theano.function(inputs=[self.input], outputs=self.h_end_star)
		##################################################################################
		# Correlated BRNN
		##################################################################################
		# Concatenate these two vectors into one
		self.h = T.concatenate([self.h_start_star, self.h_end_star], axis=0)
		# Use concatenated vector as input to the Softmax/MLP classifier
		self.output = T.nnet.softmax(T.dot(self.h, self.W_softmax) + self.b_softmax)		
		self.pred = T.argmax(self.output, axis=1)
		# Build cost function
		self.cost = -T.mean(T.log(self.output)[T.arange(self.truth.shape[0]), self.truth])
		if configs.regularization:
			self.cost += configs.lambda1 * self.L2_norm
		# Compute gradient
		self.gradtheta = T.grad(self.cost, self.theta)
		self.gradinput = T.grad(self.cost, self.input)
		# Build objective function
		# Compute the gradients to parameters
		self.compute_cost_and_gradient = theano.function(inputs=[self.input, self.truth], 
												outputs=[self.cost, self.gradtheta])
		# Compute the gradients to inputs
		self.compute_input_gradient = theano.function(inputs=[self.input, self.truth],
												outputs=self.gradinput)
		# Build prediction function
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)
		if verbose:
			pprint('*' * 50)
			pprint('Finished constructing Bidirectional Recurrent Neural Network (BRNN)')
			pprint('Size of input dimension: %d' % configs.num_input)
			pprint('Size of hidden/recurrent dimension: %d' % configs.num_hidden)
			pprint('Size of output dimension: %d' % configs.num_class)
			pprint('Is regularization applied? %s' % ('yes' if configs.regularization else 'no'))
			if configs.regularization:
				pprint('Coefficient of regularization term: %f' % configs.lambda1)
			pprint('Number of free parameters in BRNN: %d' % self.num_params)
			pprint('*' * 50)

	def train(self, input, truth, learn_rate):
		cost = self.objective(input, truth, learn_rate)
		return cost

	# This method is used to implement the batch updating algorithm
	def update_params(self, gradtheta, learn_rate):
		# gradparams is a single long vector which can be used to update self.theta
		# Learning algorithm: simple stochastic gradient descent
		theta = self.theta.get_value(borrow=True)
		self.theta.set_value(theta - learn_rate * gradtheta, borrow=True)

	@staticmethod
	def save(fname, model):
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		with file(fname, 'rb') as fin:
			return cPickle.load(fin)

