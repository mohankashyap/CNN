#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-20 09:33:13
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import numpy as np
import theano.tensor as T
import theano

from utils import floatX

class ScoreLayer(object):
    '''
    Linear Layer usually used for Regression task.
    '''
    def __init__(self, input, num_in):
        '''
        @input: theano symbolic tensor. Input to the ScoreLayer layer, a matrix
                of size (num_example, num_in), where each row of the matrix is 
                an input instance.
        @num_in: Int. Size of the input dimension.
        '''
        self.input = input
        self.W = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(num_in+1)),
                                      high=np.sqrt(6.0/(num_in+1)),
                                      size=(num_in, 1)), dtype=floatX),
                    name='W_score', borrow=True)
        self.b = theano.shared(value=np.zeros(1, dtype=floatX), name='b_score', borrow=True)
        self.output = T.dot(self.input, self.W) + self.b
        # Stack parameters
        self.params = [self.W, self.b]
        # Output score for this layer
        self.score = theano.function(inputs=[self.input], outputs=self.output)

    def L1_loss(self):
        return T.sum(T.abs_(self.W))

    def L2_loss(self):
        return T.sum(self.W ** 2)

    def encode(self, inputM):
        '''
        @inputM: Theano symbolic tensor
        '''
        return T.dot(inputM, self.W) + self.b
