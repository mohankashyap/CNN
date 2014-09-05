#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 16:02:05
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import ConfigParser

class CNNConfiger(object):
	'''
	Class for the configuration of architecture of CNN.
	'''
	def __init__(self, fname):
		'''
		@fname: String. File path to the configuration file of CNN.
		'''	
		self._cf_parser = ConfigParser.ConfigParser()
		self._cf_parser.read(fname)
		# Parsing 
		self.activation, self.nepoch, self.batch_size, self.image_row, self.image_col, \
		self.num_convpool, self.num_hidden, self.num_softmax, self.convs, self.pools, self.hiddens, \
		self.softmaxs = self.parse()

	def get(self, cfg_object, cfg_section):
		'''
		@cfg_object: String. Block title.
		@cfg_section: String. Section title.
		'''
		return self._cf_parser.get(cfg_object, cfg_section)

	def parse(self):
		activation = self._cf_parser.get('functions', 'activations')
		nepoch = self._cf_parser.getint('parameters', 'nepoch')
		batch_size = self._cf_parser.getint('input', 'batchsize')
		image_row = self._cf_parser.getint('input', 'imagerow')
		image_col = self._cf_parser.getint('input', 'imagecol')
		num_convpool = self._cf_parser.getint('architectures', 'convpool')
		num_hidden = self._cf_parser.getint('architectures', 'hidden')
		num_softmax = self._cf_parser.getint('architectures', 'softmax')
		# Load architecture of convolution and pooling layers
		convs, pools = [], []
		hiddens = []
		softmaxs = []
		# Load detailed architecture for each layer
		for i in xrange(num_convpool):
			l = self._cf_parser.get('layers', 'conv'+str(i+1))
			l = [int(x) for x in l.split(',')]
			convs.append(l)
			l = self._cf_parser.get('layers', 'pool'+str(i+1))
			l = [int(x) for x in l.split(',')]
			pools.append(l)

		for i in xrange(num_hidden):
			l = self._cf_parser.get('layers', 'hidden'+str(i+1))
			l = [int(x) for x in l.split(',')]
			hiddens.append(l)

		for i in xrange(num_softmax):
			l = self._cf_parser.get('layers', 'softmax'+str(i+1))
			l = [int(x) for x in l.split(',')]
			softmaxs.append(l)

		return (activation, nepoch, batch_size, image_row, image_col, 
				num_convpool, num_hidden, num_softmax, convs, pools, hiddens, softmaxs)


class MLPConfiger(object):
	'''
	Class for the configuration of the architecture of MLP.
	'''
	def __init__(self, fname):
		'''
		@fname: String. File path to the configuration file of MLP.
		'''
		self._cf_parser = ConfigParser.ConfigParser()
		self._cf_parser.read(fname)
		# Parsing 
		self.activation, self.nepoch, self.batch_size, self.sparsity, \
		self.lambda1, self.lambda2, self.num_hidden, self.num_softmax, \
		self.hiddens, self.softmaxs = self.parse()

	def get(self, cfg_object, cfg_section):
		'''
		@cfg_object: String. Block title.
		@cfg_section: String. Section title.
		'''
		return self._cf_parser.get(cfg_object, cfg_section)

	def parse(self):
		activation = self._cf_parser.get('functions', 'activations')
		nepoch = self._cf_parser.getint('parameters', 'nepoch')
		batch_size = self._cf_parser.getint('input', 'batchsize')
		num_hidden = self._cf_parser.getint('architectures', 'hidden')
		num_softmax = self._cf_parser.getint('architectures', 'softmax')
		
		sparsity = self._cf_parser.getint('parameters', 'sparsity')
		lambda1 = self._cf_parser.getfloat('parameters', 'lambda1')
		lambda2 = self._cf_parser.getfloat('parameters', 'lambda2')
		# Load architecture of convolution and pooling layers
		hiddens, softmaxs = [], []
		# Load detailed architecture for each layer
		for i in xrange(num_hidden):
			l = self._cf_parser.get('layers', 'hidden'+str(i+1))
			l = [int(x) for x in l.split(',')]
			hiddens.append(l)

		for i in xrange(num_softmax):
			l = self._cf_parser.get('layers', 'softmax'+str(i+1))
			l = [int(x) for x in l.split(',')]
			softmaxs.append(l)

		return (activation, nepoch, batch_size, sparsity,
				lambda1, lambda2, num_hidden, num_softmax, hiddens, softmaxs)


class DAEConfiger(object):
	'''
	Class for the configuration of the architecture of the deep [denoising|sparse] auto-encoder.
	'''
	def __init__(self, fname):
		'''
		@fname: String. File path to the configuration file of MLP.
		'''
		self._cf_parser = ConfigParser.ConfigParser()
		self._cf_parser.read(fname)
		# Parsing 
		self.activation, self.nepoch, self.seed, self.denoising, self.sparsity, \
		self.lambda1, self.mask, self.num_hidden, self.hiddens = self.parse()

	def get(self, cfg_object, cfg_section):
		'''
		@cfg_object: String. Block title.
		@cfg_section: String. Section title.
		'''
		return self._cf_parser.get(cfg_object, cfg_section)

	def parse(self):
		activation = self._cf_parser.get('functions', 'activations')
		nepoch = self._cf_parser.getint('parameters', 'nepoch')
		lambda1 = self._cf_parser.getfloat('parameters', 'lambda1')
		mask = self._cf_parser.getfloat('parameters', 'mask')
		num_hidden = self._cf_parser.getint('architectures', 'hidden')
		
		sparsity = self._cf_parser.getint('parameters', 'sparsity')
		denoising = self._cf_parser.getint('parameters', 'denoising')
		seed = self._cf_parser.getint('parameters', 'seed')
		# Load architecture of convolution and pooling layers
		hiddens = []
		# Load detailed architecture for each layer
		for i in xrange(num_hidden):
			l = self._cf_parser.get('layers', 'hidden'+str(i+1))
			l = [int(x) for x in l.split(',')]
			hiddens.append(l)

		return (activation, nepoch, seed, denoising, sparsity,
				lambda1, mask, num_hidden, hiddens)


class RNNConfiger(object):
	'''
	Class for the configuration of the architecture of RNN.
	'''
	def __init__(self, fname):
		'''
		@fname: String. File path to the configuration file of MLP.
		'''
		self._cf_parser = ConfigParser.ConfigParser()
		self._cf_parser.read(fname)
		# Parsing 
		self.activation, self.num_input, self.num_hidden, self.num_class, \
		self.regularization, self.lambda1, self.lambda2, self.bptt = self.parse()

	def get(self, cfg_object, cfg_section):
		'''
		@cfg_object: String. Block title.
		@cfg_section: String. Section title.
		'''
		return self._cf_parser.get(cfg_object, cfg_section)

	def parse(self):
		activation = self._cf_parser.get('functions', 'activations')
		num_input = self._cf_parser.getint('architectures', 'input')
		num_hidden = self._cf_parser.getint('architectures', 'hidden')
		num_class = self._cf_parser.getint('architectures', 'class')
		regularization = self._cf_parser.getint('parameters', 'regularization')
		# L1-norm regularization of the penalty function				
		lambda1 = self._cf_parser.getfloat('parameters', 'lambda1')
		# L2-norm regularization of the penalty function
		lambda2 = self._cf_parser.getfloat('parameters', 'lambda2')
		bptt = self._cf_parser.getint('parameters', 'bptt')
		return (activation, num_input, num_hidden, num_class, \
				regularization, lambda1, lambda2, bptt)

