#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-06 13:03:51
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os
import numpy as np
import theano
import theano.tensor as T
import time
import cPickle
import logging

import config
import utils
from utils import floatX
from activations import Activation

logger = logging.getLogger(__name__)

class GRCNNEncoder(object):
    '''
    (Binary) Gated Recursive Convolutional Neural Network Encoder.
    '''
    def __init__(self, config=None, verbose=True):
        '''
        @config: GRCNNConfiger. Configer used to set the architecture of GRCNNEncoder.
        ''' 
        if verbose: logger.debug('Building Gated Recursive Convolutional Neural Network Encoder...')
        # Make theano symbolic tensor for input and model parameters
        self.input = T.matrix(name='input', dtype=floatX)
        self.learn_rate = T.scalar(name='learning rate')
        # Configure activation function
        self.act = Activation(config.activation)
        fan_in, fan_out = config.num_input, config.num_hidden
        # Initialize model parameters
        # Set seed of the random generator
        np.random.seed(config.random_seed)
        # Projection matrix U
        self.U = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
                                      high=np.sqrt(6.0/(fan_in+fan_out)),
                                      size=(fan_in, fan_out)), dtype=floatX),
                    name='U', borrow=True)
        # W^l, W^r, parameters used to construct the central hidden representation
        self.Wl = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_out+fan_out)),
                                      high=np.sqrt(6.0/(fan_out+fan_out)),
                                      size=(fan_out, fan_out)), dtype=floatX),
                    name='W^l', borrow=True)
        self.Wr = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_out+fan_out)),
                                      high=np.sqrt(6.0/(fan_out+fan_out)),
                                      size=(fan_out, fan_out)), dtype=floatX),
                    name='W^r', borrow=True)
        self.Wb = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='Wb', borrow=True)
        # G^l, G^r, parameters used to construct the three-way coefficients
        self.Gl = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(3+fan_out)),
                                      high=np.sqrt(6.0/(3+fan_out)),
                                      size=(fan_out, 3)), dtype=floatX),
                    name='G^l', borrow=True)
        self.Gr = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(3+fan_out)),
                                      high=np.sqrt(6.0/(3+fan_out)),
                                      size=(fan_out, 3)), dtype=floatX),
                    name='G^r', borrow=True)
        self.Gb = theano.shared(value=np.zeros(3, dtype=floatX), name='Gb', borrow=True)
        # Save all the parameters into one batch
        self.params = [self.U, self.Wl, self.Wr, self.Wb, self.Gl, self.Gr, self.Gb]
        if verbose:
            logger.debug('Finished constructing the structure of grCNN Encoder: ')
            logger.debug('Size of the input dimension: %d' % fan_in)
            logger.debug('Size of the hidden dimension: %d' % fan_out)
            logger.debug('Activation function: %s' % config.activation)


    def _step_prop(self, current_level):
        '''
        @current_level: Input matrix at current level. The first dimension corresponds to 
        the timestamp while the second dimension corresponds to the dimension of hidden representation
        '''
        # Build shifted matrix, of size (T-1)xd
        right_current_level = current_level[1:]
        left_current_level = current_level[:-1]
        # Compute temporary central hidden representation, of size (T-1)xd
        central_current_level = self.act(T.dot(left_current_level, self.Wl) + 
                                         T.dot(right_current_level, self.Wr) + 
                                         self.Wb)
        # Compute gating function, of size (T-1)x3
        current_gates = T.nnet.softmax(T.dot(left_current_level, self.Gl) + 
                                       T.dot(right_current_level, self.Gr) + 
                                       self.Gb)
        left_gate, central_gate, right_gate = current_gates[:, 0], current_gates[:, 1], current_gates[:, 2]
        # Reshape for broadcasting
        left_gate = left_gate.dimshuffle(0, 'x')
        central_gate = central_gate.dimshuffle(0, 'x')
        right_gate = right_gate.dimshuffle(0, 'x')
        # Build next level of hidden representation using soft combination,
        # matrix of size (T-1)xd
        next_level = left_gate * left_current_level + \
                     right_gate * right_current_level + \
                     central_gate * central_current_level
        return next_level












