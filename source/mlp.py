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
from theano.tensor.shared_randomstreams import RandomStreams
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
		@configs: MLPConfiger. Configer used to set the architecture of MLP.
		'''
		if verbose: pprint('Building Multilayer Perceptron...')
		# Input setting
		self.input = T.matrix(name='input', dtype=floatX)
		self.truth = T.ivector(name='label')
		self.learn_rate = T.scalar(name='learn rate')
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
			self.updates.append((param, param-self.learn_rate*gradparam))
		# Build objective funciton
		self.objective = theano.function(inputs=[self.input, self.truth, self.learn_rate], outputs=self.cost, updates=self.updates)
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

	def train(self, batch, label, learn_rate):
		'''
		@batch: np.ndarray. Training matrix with each row as an instance.
		@label: np.ndarray. 1 dimensional array of int as labels.
		'''
		cost = self.objective(batch, label, learn_rate)
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


class AutoEncoder(object):
	'''
	Denoising Autoencoder based on fully connected hidden layers.
	'''
	def __init__(self, input, (num_in, num_out), act=None, 
				is_denoising=True, is_sparse=True, 
				lambda1=0.0, mask=0.0, rng=None, verbose=True):
		'''
		@input: theano symbolic matrix. Input to current auto-encoder.
		@num_in: np.int32. Input dimension of the auto-encoder.
		@num_out: np.int32. Output dimension of the auto-encoder.
		@act: Activation. Non-linear function applied to the neurons.
		@is_denoising: Bool. Whether to use denoising auto-encoder or not.
		@is_sparse: Bool. Whether to use sparse auto-encoder or not.
		@lambda1: floatX. If using sparse auto-encoder, the coefficient for the L2 penalty terms.
		@mask: floatX. If using denoising auto-encoder, the probability to set input elements to 0.
		@rng: theano RandomStreams. Theano random number generator.
		'''
		if verbose: pprint('Build AutoEncoder...')
		self.act = act
		self.input = input
		current_input = self.input
		if is_denoising:
			current_input *= rng.binomial(size=input.shape, n=1, p=1-mask)
		self.encode_layer = HiddenLayer(current_input, (num_in, num_out), act=self.act)
		current_input = self.encode_layer.output
		self.decode_layer = HiddenLayer(current_input, (num_out, num_in), act=self.act)
		# Build cost function
		self.cost = T.mean(T.sum((self.input-self.decode_layer.output) ** 2, axis=1))
		if is_sparse:
			self.cost += lambda1 * self.encode_layer.L2_loss()
			self.cost += lambda1 * self.decode_layer.L2_loss()
		# Compute compressed output
		self.output = self.act.activate(T.dot(self.input, self.encode_layer.W) + self.encode_layer.b)
		# Compute reconstructed output
		self.recons = self.act.activate(T.dot(self.output, self.decode_layer.W) + self.decode_layer.b)
		# Compute gradient
		self.params = []
		self.params.extend(self.encode_layer.params)
		self.params.extend(self.decode_layer.params)
		self.gradparams = T.grad(self.cost, self.params)
		# Stochastic gradient descent
		self.learn_rate = T.scalar(name='learn rate')
		self.updates = []
		for param, gradparam in zip(self.params, self.gradparams):
			self.updates.append((param, param-self.learn_rate*gradparam))
		# Build objective function
		self.objective = theano.function(inputs=[self.input, self.learn_rate], outputs=self.cost, updates=self.updates)
		# Build the compression and the reconstruction
		self._compress = theano.function(inputs=[self.encode_layer.input], outputs=self.encode_layer.output)
		self._reconstruct = theano.function(inputs=[self.decode_layer.input], outputs=self.decode_layer.output)
		# Output the building log
		if verbose:
			pprint('Architecture of AutoEncoder building finished, summarized below...')
			pprint('Is the AutoEncoder denoising: %s' % ('yes' if is_denoising else 'no'))
			pprint('Is the AutoEncoder sparse: %s' % ('yes' if is_sparse else 'no'))
			pprint('Input dimension: %d' % num_in)
			pprint('Output dimension: %d' % num_out)

	def train(self, input, learn_rate):
		'''
		@input: np.ndarray. Matrix which contains the input matrix.
		@learn_rate: floatX. Learning rate of the stochastic gradient descent algorithm.
		'''
		return self.objective(input, learn_rate)

	def compress(self, input):
		'''
		@input: np.ndarray. Matrix which contains the input matrix.
		'''
		return self._compress(input)

	def reconstruct(self, input):
		'''
		@input: np.ndarray. Matrix which contains the input matrix.
		'''
		h = self._compress(input)
		return self._reconstruct(h)

	@staticmethod
	def save(fname, model):
		'''
		@fname: String. File path to store the model.
		@model: AutoEncoder. Model of AutoEncoder to be saved.
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		@fname: String. File path to load the model.
		'''
		with file(fname, 'rb') as fin:
			model = cPickle.load(fin)
			return model


class DAE(object):
	'''
	Denoising Deep Autoencoder based on fully connected hidden layers.
	'''
	def __init__(self, configs=None, verbose=True):
		'''
		@configs: AutoEncoderConfiger. Configuration used to build the architecture of AutoEncoder.
		'''
		if verbose: pprint('Build Deep AutoEncoder...')
		# Set random number generator 
		self.rng = RandomStreams(configs.seed)
		# Symbolic input and training parameters
		self.input = T.matrix(name='input')
		self.learn_rate = T.scalar(name='learn rate')
		# Activation function
		self.act = Activation(configs.activation)
		# It may have the feature of sparse auto-encoder and also the 
		# feature of denoising auto-encoder.
		self.sparsity = configs.sparsity
		self.denoising = configs.denoising
		self.lambda1 = configs.lambda1
		self.mask = configs.mask
		# Initialize each auto-encoder layer by layer
		assert configs.num_hidden == len(configs.hiddens)
		# Stack auto-encoder
		self.autoencoders = []
		for i in xrange(configs.num_hidden):
			if i == 0: current_input = self.input
			else: current_input = self.autoencoders[i-1].output
			self.autoencoders.append(AutoEncoder(current_input, configs.hiddens[i], 
												self.act, configs.denoising, configs.sparsity, 
												configs.lambda1, configs.mask, 
												self.rng, verbose=False))
		# Stack all the parameters
		self.params = []
		for i in xrange(configs.num_hidden):
			self.params.extend(self.autoencoders[i].params)
		# Build compression 
		self.compress = theano.function(inputs=[self.input], outputs=self.autoencoders[configs.num_hidden-1].output)
		# Visualize the architecture of the deep auto-encoder
		if verbose:
			pprint('Architecture building finished, summarized below:')
			pprint('Is the deep auto-encoder sparse: %s' % ('yes' if self.sparsity else 'no'))
			pprint('Is the deep auto-encoder denoising: %s' % ('yes' if self.denoising else 'no'))
			pprint('There are %d Auto-encoders stacked.' % configs.num_hidden)
			pprint('=' * 50)
			pprint('Detailed architecture of each auto-encoder.')
			for i in xrange(configs.num_hidden):
				pprint('Auto-encoder %d, Input dimension: %d, Output dimension: %d' % (i, configs.hiddens[i][0], configs.hiddens[i][1]))

	def train(self, input, learn_rate):
		'''
		@input: np.ndarray. Two dimensional input matrix. Input to the Deep Auto-encoder.
		@learn_rate: float. Learning rate of the stochastic gradient descent algorithm.
		'''
		# Layer-wise training
		current_input = input
		total_cost = 0
		for i in xrange(len(self.autoencoders)):
			total_cost += self.autoencoders[i].train(current_input, learn_rate)
			current_input = self.autoencoders[i].compress(current_input)
		return total_cost

	@staticmethod
	def save(fname, model):
		'''
		@fname: String. File path to save DAE model.
		@model: DAE. DAE model to be saved.
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		@fname: String. File path to load DAE model.
		'''
		with file(fname, 'rb') as fin:
			model = cPickle.load(fin)
			return model

