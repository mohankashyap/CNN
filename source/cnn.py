#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 15:56:57
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
from convpool import LeNetConvPoolLayer
from mlp import HiddenLayer
from logistic import SoftmaxLayer
from activations import Activation

class ConvNet(object):
	'''
	Convolutional Neural Network based on the structure of Yann Lecun's work.
	'''
	def __init__(self, configs=None, verbose=True):
		'''
		@config: CNNConfiger. Configer used to set the architecture of CNN.
		'''
		if verbose: pprint("Building Convolutional Neural Network...")
		# Make theano symbolic tensor for input and ground truth label
		self.input = T.tensor4(name='input', dtype=floatX)
		self.truth = T.ivector(name='label')
		self.learn_rate = T.scalar(name='learn rate')
		self.batch_size = configs.batch_size
		self.image_row = configs.image_row
		self.image_col = configs.image_col
		# There may have multiple convolution-pooling and multi-layer perceptrons.
		self.convpool_layers = []
		self.mlp_layers = []
		self.softmax_layers = []
		# Configure activation function
		self.act = Activation(configs.activation)
		# Configuration should be valid
		assert configs.num_convpool == len(configs.convs)
		assert configs.num_convpool == len(configs.pools)
		assert configs.num_mlp == len(configs.mlps)
		assert configs.num_softmax == len(configs.softmaxs)
		# Build architecture of CNN
		# Convolution and Pooling layers
		image_shapes, filter_shapes = [], []
		for i in xrange(configs.num_convpool):
			if i == 0: 
				image_shapes.append((self.batch_size, 1, self.image_row, self.image_col))
				filter_shapes.append((configs.convs[i][0], 1, configs.convs[i][1], configs.convs[i][2]))
			else: 
				image_shapes.append((self.batch_size, configs.convs[i-1][0], 
							(image_shapes[i-1][2]-configs.convs[i-1][1]+1) / configs.pools[i-1][0],
							(image_shapes[i-1][3]-configs.convs[i-1][2]+1) / configs.pools[i-1][1]))
				filter_shapes.append((configs.convs[i][0], configs.convs[i-1][0], configs.convs[i][1], configs.convs[i][2]))
		for i in xrange(configs.num_convpool):
			if i == 0: 
				current_input = self.input
			else: 
				current_input = self.convpool_layers[i-1].output
			self.convpool_layers.append(LeNetConvPoolLayer(input=current_input, filter_shape=filter_shapes[i], 
					image_shape=image_shapes[i], poolsize=configs.pools[i], act=self.act))
		# Multilayer perceptron layers
		for i in xrange(configs.num_mlp):
			if i == 0: current_input = T.flatten(self.convpool_layers[configs.num_convpool-1].output, 2)
			else: current_input = self.mlp_layers[i-1].output
			self.mlp_layers.append(HiddenLayer(current_input, configs.mlps[i], act=self.act))
		# Softmax Layer, for most case, the architecture will only contain one softmax layer
		for i in xrange(configs.num_softmax):
			if i == 0: current_input = self.mlp_layers[configs.num_mlp-1].output
			else: current_input = self.softmax_layers[i-1].output
			self.softmax_layers.append(SoftmaxLayer(current_input, configs.softmaxs[i]))
		# Output
		self.pred = self.softmax_layers[configs.num_softmax-1].prediction()
		# Cost function with ground truth provided
		self.cost = self.softmax_layers[configs.num_softmax-1].NLL_loss(self.truth)
		# Build cost function 
		# Stack all the parameters
		self.params = []
		for convpool_layer in self.convpool_layers:
			self.params.extend(convpool_layer.params)
		for mlp_layer in self.mlp_layers:
			self.params.extend(mlp_layer.params)
		for softmax_layer in self.softmax_layers:
			self.params.extend(softmax_layer.params)
		# Compute gradient of self.cost with respect to network parameters
		self.gradparams = T.grad(self.cost, self.params)
		# Stochastic gradient descent learning algorithm
		self.updates = []
		for param, gradparam in zip(self.params, self.gradparams):
			self.updates.append((param, param-self.learn_rate*gradparam))
		# Build objective function
		self.objective = theano.function(inputs=[self.input, self.truth, self.learn_rate], outputs=self.cost, updates=self.updates)
		# Build prediction function
		self.predict = theano.function(inputs=[self.input], outputs=self.pred)
		if verbose:
			pprint('Architecture building finished, summarized as below: ')
			pprint('There are %d layers (not including the input layer) algether: ' % (configs.num_convpool*2 + configs.num_mlp + configs.num_softmax))
			pprint('%d convolution layers + %d maxpooling layers.' % (len(self.convpool_layers), len(self.convpool_layers)))
			pprint('%d fully connected layers.' % (len(self.mlp_layers)))
			pprint('%d softmax layers.' % (len(self.softmax_layers)))
			pprint('=' * 50)
			pprint('Detailed architecture of each layer: ')
			pprint('-' * 50)
			pprint('Convolution and Pooling layers: ')
			for i in xrange(len(self.convpool_layers)):
				pprint('Convolution Layer %d: ' % i)
				pprint('%d feature maps, each has a filter kernel with size (%d, %d)' % (configs.convs[i][0], configs.convs[i][1], configs.convs[i][2]))
			pprint('-' * 50)
			pprint('Hidden layers: ')
			for i in xrange(len(self.mlp_layers)):
				pprint('Hidden Layer %d: ' % i)
				pprint('Input dimension: %d, Output dimension: %d' % (configs.mlps[i][0], configs.mlps[i][1]))
			pprint('-' * 50)
			pprint('Softmax layers: ')
			for i in xrange(len(self.softmax_layers)):
				pprint('Softmax Layer %d: ' % i)
				pprint('Input dimension: %d, Output dimension: %d' % (configs.softmaxs[i][0], configs.softmaxs[i][1]))

	def train(self, minibatch, label, learn_rate):
		'''
		@minibatch: np.ndarray. 4th order tensor of input data, should be of type floatX.
		@label: np.ndarray. 1 dimensional array of int as labels.
		'''
		cost = self.objective(minibatch, label, learn_rate)
		pred = self.predict(minibatch)
		accuracy = np.sum(pred == label) / float(label.shape[0])
		return cost, accuracy

	@staticmethod
	def save(fname, model):
		'''
		Store current model to file
		'''
		with file(fname, 'wb') as fout:
			cPickle.dump(model, fout)

	@staticmethod
	def load(fname):
		'''
		Recover model from file
		'''
		with file(fname, 'rb') as fin:
			model = cPickle.load(fin)
			return model

