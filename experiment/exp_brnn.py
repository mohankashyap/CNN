#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 19:06:51
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import numpy as np
import theano
import theano.tensor as T
import time
import cPickle
from pprint import pprint

sys.path.append('../source/')
from rnn import RNN
from config import RNNConfiger


class BRNN(object):
	'''
	Bidirectional RNN with tied weights. This is just a trial for using 
	BRNN as a tool for sentence modeling.

	First trial on the task of sentiment analysis.
	'''
	def __init__(self, configs, verbose=True):
		if verbose: pprint('Build ')