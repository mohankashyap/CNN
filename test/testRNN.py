#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-29 17:54:41
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import unittest
import numpy as np
import theano
import theano.tensor as T
from pprint import pprint
sys.path.append('../')

from rnn import RNN
from config import RNNConfiger

