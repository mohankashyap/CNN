#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 18:11:04
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import numpy as np
import theano
import theano.tensor as T
import utils

class Activation(object):
	def __init__(self, method):
		'''
		@method: String. Method to use as the activation function for each neuron.
		'''
		if method == "sigmoid":
			self.func = T.nnet.sigmoid
		elif method == "tanh":
			self.func = T.tanh
		elif method == "ReLU":
			self.func = utils.reLU
		else:
			raise ValueError('Invalid Activation function!')

	def activate(self, x):
		'''
		@x: theano symbolic tensor. Object to be applied with the chosen
			activation function.
		'''
		return self.func(x)