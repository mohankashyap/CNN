#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-04 10:38:30
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Version : 0.0

import os, sys
import numpy as np
import theano
import theano.tensor as T

floatX = theano.config.floatX

def loadtxt(fname):
	'''
	@fname: String. File path to the snippet-[train|test].txt file.
	'''
	snippets = list()
	with file(fname, 'r') as fin:
		for line in fin:
			snippets.append(line.split())
	return snippets

def loadlabel(fname):
	'''
	@fname: String. File path to the snippet-[train|test]-label.txt file
	'''
	labels = list()
	with file(fname, 'r') as fin:
		labels = fin.readlines()
	labels = np.array([int(x.strip()) for x in labels])
	return labels

def reLU(x):
	'''
	@x:	theano symbolic tensor. Activation function used at each neuron in 
		hidden layer. Proposed by Hinton et al.
	'''
	return T.maximum(0, x)

