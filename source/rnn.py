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
import logging
from pprint import pprint

import config
import utils
from utils import floatX
from activations import Activation
from logistic import SoftmaxLayer, LogisticLayer
from mlp import HiddenLayer

logger = logging.getLogger(__name__)

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
		self.hs, _ = theano.scan(fn=step, sequences=self.input, outputs_info=[self.h0],
								truncate_gradient=configs.bptt)
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
		self.forward_h, _ = theano.scan(fn=forward_step, sequences=self.input, outputs_info=[self.h_start],
										truncate_gradient=configs.bptt)
		self.backward_h, _ = theano.scan(fn=backward_step, sequences=self.input, outputs_info=[self.h_end], 
										 truncate_gradient=configs.bptt, go_backwards=True)
		# Store the final value
		# self.h_start_star = self.forward_h[-1]
		# self.h_end_star = self.backward_h[-1]
		self.h_start_star = T.mean(self.forward_h, axis=0)
		self.h_end_star = T.mean(self.backward_h, axis=0)
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
		# Dropout parameter
		srng = T.shared_randomstreams.RandomStreams(configs.random_seed)
		mask = srng.binomial(n=1, p=1-configs.dropout, size=self.h.shape)
		self.h *= T.cast(mask, floatX)
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
			pprint('BPTT step: %d' % configs.bptt)
			pprint('Number of free parameters in BRNN: %d' % self.num_params)
			pprint('*' * 50)

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


class BRNNEncoder(object):
	'''
	Bidirectional RNN for sequence encoding. 
	'''
	def __init__(self, config, verbose=True):
		if verbose: logger.debug('Building Bidirectional RNN Encoder...')
		self.input = T.matrix(name='BRNNEncoder_input')
		# Configure Activation function
		self.act = Activation(config.activation)
		# Build Bidirectional RNN
		num_input, num_hidden = config.num_input, config.num_hidden
		self.num_params = 2 * (num_input * num_hidden + num_hidden * num_hidden + num_hidden)
		# Initialize model parameters
		np.random.seed(config.random_seed)
		# 1, Feed-forward matrix for forward direction: W_forward
		W_forward_val = np.random.uniform(low=-1.0, high=1.0, size=(num_input, num_hidden))
		W_forward_val = W_forward_val.astype(floatX)
		self.W_forward = theano.shared(value=W_forward_val, name='W_forward', borrow=True)
		# 1, Feed-forward matrix for backward direction: W_backward
		W_backward_val = np.random.uniform(low=-1.0, high=1.0, size=(num_input, num_hidden))
		W_backward_val = W_backward_val.astype(floatX)
		self.W_backward = theano.shared(value=W_backward_val, name='W_backward', borrow=True)
		# 2, Recurrent matrix for forward direction: U_forward
		U_forward_val = np.random.uniform(low=-1.0, high=1.0, size=(num_hidden, num_hidden))
		U_forward_val = U_forward_val.astype(floatX)
		U_forward_val, _, _ = np.linalg.svd(U_forward_val)
		self.U_forward = theano.shared(value=U_forward_val, name='U_forward', borrow=True)
		# 2, Recurrent matrix for backward direction: U_backward
		U_backward_val = np.random.uniform(low=-1.0, high=1.0, size=(num_hidden, num_hidden))
		U_backward_val = U_backward_val.astype(floatX)
		U_backward_val, _, _ = np.linalg.svd(U_backward_val)
		self.U_backward = theano.shared(value=U_backward_val, name='U_backward', borrow=True)
		# 3, Bias parameter for the hidden-layer forward direction RNN
		b_forward_val = np.zeros(num_hidden, dtype=floatX)
		self.b_forward = theano.shared(value=b_forward_val, name='b_forward', borrow=True)
		# 3, Bias parameter for the hidden-layer backward direction RNN
		b_backward_val = np.zeros(num_hidden, dtype=floatX)
		self.b_backward = theano.shared(value=b_backward_val, name='b_backward', borrow=True)
		# h[0], zero vectors, treated as constants
		self.h0_forward = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h0_forward', borrow=True)
		self.h0_backward = theano.shared(value=np.zeros(num_hidden, dtype=floatX), name='h0_backward', borrow=True)
		# Stack all the parameters
		self.params = [self.W_forward, self.W_backward, self.U_forward, self.U_backward, 
					   self.b_forward, self.b_backward]
		# Compute the forward and backward representation over time
		self.h_forwards, _ = theano.scan(fn=self._forward_step, 
										 sequences=self.input, 
										 outputs_info=[self.h0_forward],
										 truncate_gradient=config.bptt)
		self.h_backwards, _ = theano.scan(fn=self._backward_step,
										  sequences=self.input,
										  outputs_info=[self.h0_backward],
										  truncate_gradient=config.bptt,
										  go_backwards=True)
		# Average compressing
		self.h_forward = T.mean(self.h_forwards, axis=0)
		self.h_backward = T.mean(self.h_backwards, axis=0)
		# Concatenate
		self.output = T.concatenate([self.h_forward, self.h_backward], axis=1)
		# L1, L2 regularization
		self.L1_norm = T.sum(T.abs_(self.W_forward) + T.abs_(self.W_backward) + 
							 T.abs_(self.U_forward) + T.abs_(self.U_backward))
		self.L2_norm = T.sum(self.W_forward ** 2) + T.sum(self.W_backward ** 2) + \
					   T.sum(self.U_forward ** 2) + T.sum(self.U_backward ** 2)
		if verbose:
			logger.debug('Finished constructing the structure of BRNN Encoder: ')
			logger.debug('Size of the input dimension: %d' % num_input)
			logger.debug('Size of the hidden dimension: %d' % num_hidden)
			logger.debug('Activation function: %s' % config.activation)

	def _forward_step(self, x_t, h_tm1):
		h_t = self.act.activate(T.dot(x_t, self.W_forward) + \
								T.dot(h_tm1, self.U_forward) + \
								self.b_forward)
		return h_t

	def _backward_step(self, x_t, h_tm1):
		h_t = self.act.activate(T.dot(x_t, self.W_backward) + \
								T.dot(h_tm1, self.U_backward) + \
								self.b_backward)
		return h_t				

	def encode(self, inputM):
		'''
		@inputM: Theano symbol matrix. Compress the input matrix into output vector.
		'''
		h_forwards, _ = theano.scan(fn=self._forward_step, 
									sequences=inputM,
									outputs_info=[self.h0_forward])
		h_backwards, _ = theano.scan(fn=self._backward_step, 
									 sequences=inputM,
									 outputs_info=[self.h0_backward],
									 go_backwards=True)
		# Averaging
		h_forward = T.mean(h_forwards, axis=0)
		h_backward = T.mean(h_backwards, axis=0)
		# Concatenate
		h = T.concatenate([h_forward, h_backward], axis=0)
		return h


class BRNNMatcher(object):
	'''
	Bidirectional RNN for text matching as a classification problem.
	'''
	def __init__(self, config, verbose=True):
		# Construct two BRNNEncoders for matching two sentences
		self.encoderL = BRNNEncoder(config, verbose)
		self.encoderR = BRNNEncoder(config, verbose)
		# Link two parts
		self.params = []
		self.params += self.encoderL.params
		self.params += self.encoderR.params
		# Set up input
		self.inputL = self.encoderL.input
		self.inputR = self.encoderR.input
		# Get output of two BRNNEncoders
		self.hiddenL = self.encoderL.output
		self.hiddenR = self.encoderR.output
		# Activation function
		self.act = Activation(config.activation)
		# MLP Component
		self.hidden = T.concatenate([self.hiddenL, self.hiddenR], axis=0)
		self.hidden_layer = HiddenLayer(self.hidden, 
										(4*config.num_hidden, config.num_mlp), 
										act=Activation(config.hiddenact))
		self.compressed_hidden = self.hidden_layer.output
		# Accumulate parameters
		self.params += self.hidden_layer.params
		# Dropout parameter
		srng = T.shared_randomstreams.RandomStreams(config.random_seed)
		mask = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hidden.shape)
		self.compressed_hidden *= T.cast(mask, floatX)
		# Logistic regression
		self.logistic_layer = LogisticLayer(self.compressed_hidden, config.num_mlp)
		self.output = self.logistic_layer.output
		self.pred = self.logistic_layer.pred
		# Accumulate parameters
		self.params += self.logistic_layer.params
		# Compute the total number of parameters in the model
		self.num_params_encoder = self.encoderL.num_params + self.encoderR.num_params
		self.num_params_classifier = 2 * config.num_hidden * config.num_mlp + config.num_mlp + \
									 config.num_mlp + 1
		self.num_params = self.num_params_encoder + self.num_params_classifier
		# Build target function
		self.truth = T.ivector(name='label')
		self.cost = self.logistic_layer.NLL_loss(self.truth)
		# Build computational graph and compute the gradients of the model parameters
		# with respect to the cost function
		self.gradparams = T.grad(self.cost, self.params)
		# Compile theano function
		self.objective = theano.function(inputs=[self.inputL, self.inputR, self.truth], outputs=self.cost)
		self.predict = theano.function(inputs=[self.inputL, self.inputR], outputs=self.pred)
		# Compute the gradient of the objective function and cost and prediction
		self.compute_cost_and_gradient = theano.function(inputs=[self.inputL, self.inputR, self.truth],
														 outputs=self.gradparams+[self.cost, self.pred])
		# Output function for debugging purpose
		self.show_hidden = theano.function(inputs=[self.inputL, self.inputR], outputs=self.hidden)
		self.show_compressed_hidden = theano.function(inputs=[self.inputL, self.inputR], outputs=self.compressed_hidden)
		self.show_output = theano.function(inputs=[self.inputL, self.inputR], outputs=self.output)
		if verbose:
			logger.debug('Architecture of BRNNMatcher built finished, summarized below: ')
			logger.debug('Input dimension: %d' % config.num_input)
			logger.debug('Hidden dimension of RNN: %d' % config.num_hidden)
			logger.debug('Hidden dimension of MLP: %d' % config.num_mlp)
			logger.debug('Number of parameters in the encoder part: %d' % self.num_params_encoder)
			logger.debug('Number of parameters in the classifier: %d' % self.num_params_classifier)
			logger.debug('Total number of parameters in this model: %d' % self.num_params)

	def update_params(self, grads, learn_rate):
		'''
		@grads: [np.ndarray]. List of numpy.ndarray for updating the model parameters.
				They are the corresponding gradients of model parameters.
		@learn_rate: scalar. Learning rate.
		'''
		for param, grad in zip(self.params, grads):
			p = param.get_value(borrow=True)
			param.set_value(p - learn_rate * grad, borrow=True)

	@staticmethod
	def save(fname, model):
		'''
		@fname: String. Filename to store the model.
		@model: BRNNMatcher. An instance of BRNNMatcher to be saved.
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		@fname: String. Filename to load the model.
		'''
		with file(fname, 'rb') as fin:
			model = cPickle.load(fin)
		return model


class BRNNMatchScorer(object):
	'''
	Bidirectional RNN for text matching as a classification problem.
	'''
	def __init__(self, config, verbose=True):
		# Construct two BRNNEncoders for matching two sentences
		self.encoderL = BRNNEncoder(config, verbose)
		self.encoderR = BRNNEncoder(config, verbose)
		# Link two parts
		self.params = []
		self.params += self.encoderL.params
		self.params += self.encoderR.params
		# Set up input
		# Note that there are three kinds of inputs altogether, including:
		# 1, inputL, inputR. This pair is used for computing the score after training
		# 2, inputPL, inputPR. This pair is used for training positive pairs
		# 3, inputNL, inputNR. This pair is used for training negative pairs
		self.inputL = self.encoderL.input
		self.inputR = self.encoderR.input
		# Positive 
		self.inputPL = T.matrix(name='inputPL', dtype=floatX)
		self.inputPR = T.matrix(name='inputPR', dtype=floatX)
		# Negative
		self.inputNL = T.matrix(name='inputNL', dtype=floatX)
		self.inputNR = T.matrix(name='inputNR', dtype=floatX)
		# Get output of two BRNNEncoders
		self.hiddenL = self.encoderL.output
		self.hiddenR = self.encoderR.output
		# Positive Hidden
		self.hiddenPL = self.encoderL.encode(self.inputPL)
		self.hiddenPR = self.encoderR.encode(self.inputPR)
		# Negative Hidden
		self.hiddenNL = self.encoderL.encode(self.inputNL)
		self.hiddenNR = self.encoderR.encode(self.inputNR)
		# Activation function
		self.act = Activation(config.activation)
		self.hidden = T.concatenate([self.hiddenL, self.hiddenR], axis=1)
		self.hiddenP = T.concatenate([self.hiddenPL, self.hiddenPR], axis=1)
		self.hiddenN = T.concatenate([self.hiddenNL, self.hiddenNR], axis=1)
		# Build hidden layer
		self.hidden_layer = HiddenLayer(self.hidden, 
										(4*config.num_hidden, config.num_mlp), 
										act=Activation(config.hiddenact))
		self.compressed_hidden = self.hidden_layer.output
		self.compressed_hiddenP = self.hidden_layer.encode(self.hiddenP)
		self.compressed_hiddenN = self.hidden_layer.encode(self.hiddenN)
		# Accumulate parameters
		self.params += self.hidden_layer.params
		# Dropout parameter
		srng = T.shared_randomstreams.RandomStreams(config.random_seed)
		mask = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hidden.shape)
		maskP = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hiddenP.shape)
		maskN = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hiddenN.shape)
		self.compressed_hidden *= T.cast(mask, floatX)
		self.compressed_hiddenP *= T.cast(maskP, floatX)
		self.compressed_hiddenN *= T.cast(maskN, floatX)
		# Score layer
		self.score_layer = ScoreLayer(self.compressed_hidden, config.num_mlp)
		self.output = self.score_layer.output
		self.scoreP = self.score_layer.encode(self.compressed_hiddenP)
		self.scoreN = self.score_layer.encode(self.compressed_hiddenN)
		# Accumulate parameters
		self.params += self.score_layer.params
		# Build cost function
		self.cost = T.mean(T.maximum(T.zero_likes(self.scoreP), 1.0 - self.scoreP + self.scoreN))
		# Construct the total number of parameters in the model
		self.gradparams = T.grad(self.cost, self.params)
		# Compute the total number of parameters in the model
		self.num_params_encoder = self.encoderL.num_params + self.encoderR.num_params
		self.num_params_classifier = 2 * config.num_hidden * config.num_mlp + config.num_mlp + \
									 config.num_mlp + 1
		self.num_params = self.num_params_encoder + self.num_params_classifier
		# Build class functions
		self.score = theano.function(inputs=[self.inputL, self.inputR], outputs=self.output)
		# Compute the gradient of the objective function and cost and prediction
		self.compute_cost_and_gradient = theano.function(inputs=[self.inputPL, self.inputPR, 
																 self.inputNL, self.inputNR],
														 outputs=self.gradparams+[self.cost, self.scoreP, self.scoreN])
		# Output function for debugging purpose
		self.show_scores = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR],
										   outputs=[self.scoreP, self.scoreN])
		self.show_hiddens = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR],
											outputs=[self.hiddenP, self.hiddenN])
		if verbose:
			logger.debug('Architecture of BRNNMatchScorer built finished, summarized below: ')
			logger.debug('Input dimension: %d' % config.num_input)
			logger.debug('Hidden dimension of RNN: %d' % config.num_hidden)
			logger.debug('Hidden dimension of MLP: %d' % config.num_mlp)
			logger.debug('There are 2 BRNNEncoders used in the model.')
			logger.debug('Total number of parameters in this model: %d' % self.num_params)

	def update_params(self, grads, learn_rate):
		'''
		@grads: [np.ndarray]. List of numpy.ndarray for updating the model parameters.
				They are the corresponding gradients of model parameters.
		@learn_rate: scalar. Learning rate.
		'''
		for param, grad in zip(self.params, grads):
			p = param.get_value(borrow=True)
			param.set_value(p - learn_rate * grad, borrow=True)

	def set_params(self, params):
		'''
		@params: [np.ndarray]. List of numpy.ndarray to set the model parameters.
		'''
		for p, param in zip(self.params, params):
			p.set_value(param, borrow=True)

	def deepcopy(self, brnn):
		'''
		@brnn: BRNNMatchScorer. Copy the model parameters of another BRNNMatchScorer.
		'''
		assert len(self.params) == len(brnn.params)
		for p, param in zip(self.params, brnn.params):
			val = param.get_value()
			p.set_value(val)

	@staticmethod
	def save(fname, model):
		'''
		@fname: String. Filename to store the model.
		@model: BRNNMatcher. An instance of BRNNMatcher to be saved.
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		@fname: String. Filename to load the model.
		'''
		with file(fname, 'rb') as fin:
			model = cPickle.load(fin)
		return model


