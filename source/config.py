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
		self.activation, self.learning_rate, self.nepoch, self.batch_size, self.image_row, self.image_col, \
		self.num_convpool, self.num_mlp, self.num_softmax, self.convs, self.pools, self.mlps, \
		self.softmaxs = self.parse()

	def get(self, cfg_object, cfg_section):
		'''
		@cfg_object: String. Block title.
		@cfg_section: String. Section title.
		'''
		return self._cf_parser.get(cfg_object, cfg_section)

	def parse(self):
		activation = self._cf_parser.get('functions', 'activations')
		learning_rate = self._cf_parser.getfloat('parameters', 'learnrate')
		nepoch = self._cf_parser.getint('parameters', 'nepoch')
		batch_size = self._cf_parser.getint('input', 'batchsize')
		image_row = self._cf_parser.getint('input', 'imagerow')
		image_col = self._cf_parser.getint('input', 'imagecol')
		num_convpool = self._cf_parser.getint('architectures', 'convpool')
		num_mlp = self._cf_parser.getint('architectures', 'mlp')
		num_softmax = self._cf_parser.getint('architectures', 'softmax')
		# Load architecture of convolution and pooling layers
		convs, pools = [], []
		mlps = []
		softmaxs = []
		# Load detailed architecture for each layer
		for i in xrange(num_convpool):
			l = self._cf_parser.get('layers', 'conv'+str(i+1))
			l = [int(x) for x in l.split(',')]
			convs.append(l)
			l = self._cf_parser.get('layers', 'pool'+str(i+1))
			l = [int(x) for x in l.split(',')]
			pools.append(l)

		for i in xrange(num_mlp):
			l = self._cf_parser.get('layers', 'mlp'+str(i+1))
			l = [int(x) for x in l.split(',')]
			mlps.append(l)

		for i in xrange(num_softmax):
			l = self._cf_parser.get('layers', 'softmax'+str(i+1))
			l = [int(x) for x in l.split(',')]
			softmaxs.append(l)

		return (activation, learning_rate, nepoch, batch_size, image_row, image_col, 
				num_convpool, num_mlp, num_softmax, convs, pools, mlps, softmaxs)



