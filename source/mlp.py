#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 14:51:23
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import utils
import numpy as np
import cPickle
import theano
import theano.tensor as T
from activations import Activation
from logistic import SoftmaxLayer
from pprint import pprint
from utils import floatX

class HiddenLayer(object):
	'''
	Hidden layer of Multilayer Perceptron
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
									  size=(num_in, num_out)), dtype=floatX),
					name='W', borrow=True)
		self.b = theano.shared(value=np.zeros(num_out, dtype=floatX), name='b', borrow=True)
		self.output = act.activate(T.dot(self.input, self.W) + self.b)
		# Stack parameters
		self.params = [self.W, self.b]

	def L2_loss(self):
		return T.sum(self.W ** 2)


class MLP(object):
	'''
	Multilayer Perceptron
	'''
	def __init__(self, configs=None, verbose=True):
		'''
		@config: MLPConfiger. Configer used to set the architecture of MLP.
		'''
		if verbose: pprint('Building Multilayer Perceptron...')
		self.input = T.matrix(name='input', dtype=floatX)
		self.truth = T.ivector(name='label')
		self.batch_size = configs.batch_size
		# There may have multiple hidden layers and softmax layers
		self.hidden_layers = []
		self.softmax_layers = []
		# Configure activation function
		self.act = Activation(configs.activation)
		# Configuration should be valid
		assert configs.num_hidden == len(configs.hiddens)
		assert configs.num_softmax == len(configs.softmaxs)
		# Build architecture of MLP
		# Build hidden layers
		for i in xrange(configs.num_hidden):
			if i == 0: current_input = self.input
			else: current_input = self.hidden_layers[i-1].output
			self.hidden_layers.append(HiddenLayer(current_input, configs.hiddens[i], act=self.act))
		# Build softmax layers
		for i in xrange(configs.num_softmax):
			if i == 0: current_input = self.hidden_layers[configs.num_hidden-1].output
			else: current_input = self.softmax_layers[i-1].output
			self.softmax_layers.append(SoftmaxLayer(current_input, configs.softmaxs[i]))
		# Output
		self.pred = self.softmax_layers[configs.num_softmax-1].prediction()
		# Cost function with ground truth provided
		self.cost = self.softmax_layers[configs.num_softmax-1].NLL_loss(self.truth)
		# Force weight matrix to be sparse
		if configs.sparsity: 
			for i in xrange(configs.num_hidden):
				self.cost += configs.lambda1 * self.hidden_layers[i].L2_loss()
			for i in xrange(configs.num_softmax):
				self.cost += configs.lambda2 * self.softmax_layers[i].L2_loss()
		# Stack all the parameters
		self.params = []
		for hidden_layer in self.hidden_layers:
			self.params.extend(hidden_layer.params)
		for softmax_layer in self.softmax_layers:
			self.params.extend(softmax_layer.params)
		# Compute gradient vector with respect to network parameters
		self.gradparams = T.grad(self.cost, self.params)
		# Stochastic gradient descent
		self.updates = []
		for param, gradparam in zip(self.params, self.gradparams):
			self.updates.append((param, param-configs.learning_rate*gradparam))
		# Build objective funciton
		self.objective = theano.function(inputs=[self.input, self.truth], outputs=self.cost, updates=self.updates)
		# Build prediction function
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)
		if verbose:
			pprint('Architecture building finished, summarized as below: ')
			pprint('There are %d layers (not including the input layer) altogether: ' % (configs.num_hidden + configs.num_softmax))
			pprint('%d hidden layers.' % configs.num_hidden)
			pprint('%d softmax layers.' % configs.num_softmax)
			pprint('=' * 50)
			pprint('Detailed architecture of each layer: ')
			pprint('-' * 50)
			pprint('Hidden layers: ')
			for i in xrange(configs.num_hidden):
				pprint('Hidden Layer: %d' % i)
				pprint('Input dimension: %d, Output dimension: %d' % (configs.hiddens[i][0], configs.hiddens[i][1]))
			for i in xrange(configs.num_softmax):
				pprint('Softmax Layer: %d' % i)
				pprint('Input dimension: %d, Output dimension: %d' % (configs.softmaxs[i][0], configs.softmaxs[i][1]))

	def train(self, batch, label):
		'''
		@batch: np.ndarray. Training matrix with each row as an instance.
		@label: np.ndarray. 1 dimensional array of int as labels.
		'''
		cost = self.objective(batch, label)
		pred = self.predict(batch)
		accuracy = np.sum(pred == label) / float(label.shape[0])
		return cost, accuracy

	@staticmethod
	def save(fname, model):
		'''
		@fname: String. Filename to store the model.
		@model: MLP. An instance of MLP to be saved.
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

