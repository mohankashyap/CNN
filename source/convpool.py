#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 11:47:52
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao

import os, sys
import numpy as np
import utils
import theano
import theano.tensor as T
from pprint import pprint
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from utils import floatX

class LeNetConvPoolLayer(object):
	'''
	Convolution layer and Pooling layer used in CNN
	'''
	def __init__(self, input=None, filter_shape=None, image_shape=None, poolsize=None, act=None):
		'''
		@input: theano symbolic tensor. Input to current Covolution and Pooling
				layer, a 4th order tensor.

		@filter_shape: tuple/list. Shape of the filters used in LeNetConvPoolLayer, 
				of value: (num_of_feature_maps, num_of_prev_feature_maps, filter_row, filter_col), 
				for one feature_map in this layer, it's a 3D tensor with shape (num_of_prev_feature_maps, 
				filter_row, filter_col)

		@image_shape: tuple/list. Shape of the images used as input to LeNetConvPoolLayer,
				of value: (batch_size, num_feature_maps, image_row, image_col)
		
		@poolsize: tuple/list. Shape of pooling kernel, of value: (pool_row, pool_col)
		@act: Activation. Activation function used for each neuron.
		'''
		assert filter_shape[1] == image_shape[1]
		# Initialization according to a common trick	
		fan_in = np.prod(filter_shape[1:])
		self.input = input
		self.W = theano.shared(value=np.asarray(
					np.random.uniform(low=-np.sqrt(6.0/fan_in),
									  high=np.sqrt(6.0/fan_in),
									  size=filter_shape), dtype=floatX),
					name='W', borrow=True)
		self.b = theano.shared(value=np.zeros(filter_shape[0], dtype=floatX),
					name='b', borrow=True)
		conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
		pooled_out = downsample.max_pool_2d(conv_out, ds=poolsize, ignore_border=True)
		self.output = act.activate(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		# Stack parameters
		self.params = [self.W, self.b]

