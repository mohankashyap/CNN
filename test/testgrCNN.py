#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-06 17:19:06
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import unittest
import theano
import theano.tensor as T
import numpy as np
import logging
import pprint
import time

# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
sys.path.append('../source/')
logger = logging.getLogger(__name__)

theano.config.exception_verbosity='high'

from grcnn import GrCNNEncoder
from config import GrCNNConfiger

class TestGRCNN(unittest.TestCase):
    def setUp(self):
        fname = './grCNN.conf'
        self.configer = GrCNNConfiger(fname)

    @unittest.skip('Finished building, no problem')
    def testBuilding(self):
        logger.debug('Inside testBuilding...')
        grcnn = GrCNNEncoder(self.configer, verbose=True)

    @unittest.skip('One Step propagation function passed.')
    def testOneStep(self):
        logger.debug('Inside testOneStep...')
        grcnn = GrCNNEncoder(self.configer, verbose=True)
        time_steps = 30000
        input_matrix = np.random.rand(time_steps, 10)
        input_matrix = input_matrix.astype(np.float32)
        logger.debug('Input Matrix Shape: {}'.format(input_matrix.shape))
        # Manually compute the result
        U = grcnn.U.get_value(borrow=True)
        Wl = grcnn.Wl.get_value(borrow=True)
        Wr = grcnn.Wr.get_value(borrow=True)
        Wb = grcnn.Wb.get_value(borrow=True)
        Gl = grcnn.Gl.get_value(borrow=True)
        Gr = grcnn.Gr.get_value(borrow=True)
        Gb = grcnn.Gb.get_value(borrow=True)
        # Build function
        input_symbol = T.matrix()
        hidden_symbol = T.dot(input_symbol, U)
        output_symbol = grcnn._step_prop(hidden_symbol)
        f = theano.function(inputs=[input_symbol], outputs=output_symbol)
        # Start testing
        start_time = time.time()
        output_matrix = f(input_matrix)
        end_time = time.time()
        logger.debug('Time used by Theano: %f seconds.' % (end_time-start_time))
        def softmax(x):
            y = np.exp(x)
            return y / np.sum(y, axis=1)[:, np.newaxis]
        start_time = time.time()
        hidden_matrix = np.dot(input_matrix, U)
        left_T, right_T = hidden_matrix[:-1], hidden_matrix[1:]
        central_T = np.tanh(np.dot(left_T, Wl) + np.dot(right_T, Wr) + Wb)
        gates = softmax(np.dot(left_T, Gl) + np.dot(right_T, Gr) + Gb)
        left_gate, central_gate, right_gate = gates[:, 0], gates[:, 1], gates[:, 2]
        next_level = left_T * left_gate[:, np.newaxis] + \
                     right_T * right_gate[:, np.newaxis] + \
                     central_T * central_gate[:, np.newaxis]
        end_time = time.time()
        logger.debug('Time used by Numpy: %f seconds.' % (end_time-start_time))
        diff1 = np.sum(np.abs(next_level-output_matrix[:-1]))
        logger.debug('L1 difference: {}'.format(diff1))
        diff2 = np.sum((next_level-output_matrix[:-1]) ** 2)
        logger.debug('L2 difference: {}'.format(diff2))

    # @unittest.skip('Wait a minute')
    def testCompress(self):
        logger.debug('Inside testCompress...')
        grcnn = GrCNNEncoder(self.configer, verbose=True)
        # Build step transformation function 
        input_symbol = T.matrix()
        output_symbol = grcnn._step_prop_reduce(input_symbol)
        f = theano.function(inputs=[input_symbol], outputs=output_symbol)
        time_steps = 600
        input_matrix = np.random.rand(time_steps, self.configer.num_input)
        input_matrix = input_matrix.astype(np.float32)
        # Timing for numpy iterative application
        start_time = time.time()
        U = grcnn.U.get_value(borrow=True)
        hidden_matrix = np.dot(input_matrix, U)
        logger.debug('Input Matrix Shape: {}'.format(hidden_matrix.shape))
        for i in xrange(time_steps-1):
            hidden_matrix = f(hidden_matrix)
        end_time = time.time()
        logger.debug('Time used for manually iterative application: {} seconds.'.format(end_time-start_time))
        # logger.debug('Output Matrix by manually iterative application: ')
        # logger.debug(hidden_matrix)
        logger.debug('Output Matrix Shape: {}'.format(hidden_matrix.shape))
        logger.debug('=' * 50)
        start_time = time.time()
        output_matrix = grcnn.compress(input_matrix)
        end_time = time.time()
        logger.debug('Time used for Theano.scan implementation: {} seconds.'.format(end_time-start_time))
        logger.debug('Output Matrix Shape: {}'.format(output_matrix.shape))
        # logger.debug('Output Matrix by theano.scan compress function: ')
        # logger.debug(output_matrix)
        diff1 = np.sum(np.abs(hidden_matrix-output_matrix))
        diff2 = np.sum((hidden_matrix-output_matrix) ** 2)
        logger.debug('*' * 50)
        logger.debug('L1 difference between manually application and Theano compression: {}'.format(diff1))
        logger.debug('L2 difference between manually application and Theano compression: {}'.format(diff2))

if __name__ == '__main__':
    unittest.main()

