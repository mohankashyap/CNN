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
import logging

sys.path.append('../source/')
logger = logging.getLogger(__name__)

from grcnn import GRCNNEncoder
from config import GRCNNConfiger

class TestGRCNN(unittest.TestCase):
    def setUp(self):
        fname = './grCNN.conf'
        self.configer = GRCNNConfiger(fname)

    def testBuilding(self):
        grcnn = GRCNNEncoder(self.configer, verbose=True)



if __name__ == '__main__':
    unittest.main()

