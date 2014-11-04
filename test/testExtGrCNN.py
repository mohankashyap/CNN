#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-11-04 16:36:47
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import theano
import theano.tensor as T
import numpy as np
import scipy as sp
import unittest
import logging
import time

sys.path.append('../source')

from config import GrCNNConfiger
from grcnn import ExtGrCNNEncoder

# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

class TestExtGrCNN(unittest.TestCase):
    def setUp(self):
        fname = './extgrcnn.conf'
        self.configer = GrCNNConfiger(fname)

    @unittest.skip('Passed...')
    def testBuild(self):
        logger.debug('Inside testBuild...')
        grcnn = ExtGrCNNEncoder(self.configer)

    def testConsistency(self):
        logger.debug('Inside test Consistency...')
        grcnn = ExtGrCNNEncoder(self.configer)
        params = [param.get_value(borrow=True) for param in grcnn.params]
        logger.debug('Model parameter: ')
        for param in params:
            logger.debug(param.shape)
        inputM = np.random.rand(40, 50).astype(np.float32)
        start_time = time.time()
        outputM = grcnn.compress(inputM)
        end_time = time.time()
        logger.debug('Time used to compress for theano: %f seconds.' % (end_time-start_time))
        # Manually computing to ensure the computation done by theano is correct
        param_U, param_Wl, param_Wr, param_Wb, param_Gl, param_Gr, param_Gb = params
        start_time = time.time()
        hiddenM = np.dot(inputM, param_U)
        nsteps = inputM.shape[0]
        for i in xrange(nsteps-1):
            left_hiddenM = hiddenM[:-1]
            right_hiddenM = hiddenM[1:]
            multi_centrals = np.tanh(np.dot(left_hiddenM, param_Wl) + 
                                     np.dot(right_hiddenM, param_Wr) +
                                     param_Wb)
            multi_gates = np.dot(left_hiddenM, param_Gl) + \
                          np.dot(right_hiddenM, param_Gr) + \
                          param_Gb
            multi_gates = np.exp(multi_gates)
            multi_gates /= np.sum(multi_gates, axis=1)[:, np.newaxis]
            padded = np.zeros((hiddenM.shape[0]-1, multi_centrals.shape[1]+2, 
                            multi_centrals.shape[2]))
            padded[:, 1:-1, :] = multi_centrals
            for j in xrange(padded.shape[0]):
                padded[j, 0, :] = left_hiddenM[j, :]
                padded[j, -1, :] = right_hiddenM[j, :]
            multi_centrals = padded
            multi_gates = multi_gates[:, :, np.newaxis]
            hiddenM = multi_gates * multi_centrals
            hiddenM = np.sum(hiddenM, axis=1)
        myoutputM = hiddenM
        end_time = time.time()
        logger.debug('Time used to compress for numpy: %f seconds. ' % (end_time-start_time))
        L1_diff = np.sum(np.abs(myoutputM - outputM))
        L2_diff = np.sum(np.square(myoutputM - outputM))
        logger.debug('L1 difference: {}'.format(L1_diff))
        logger.debug('L2 difference: {}'.format(L2_diff))
        logger.debug('Theano output: ')
        logger.debug(outputM)
        logger.debug('Numpy output: ')
        logger.debug(myoutputM)

if __name__ == '__main__':
    unittest.main()



