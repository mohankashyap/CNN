#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-14 20:04:03
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import numpy as np
import theano
import theano.tensor as T
import scipy
import scipy.io as sio
import unittest
import csv
import time
import cPickle
import logging

from pprint import pprint

sys.path.append('../source/')

from rnn import BRNN, TBRNN, RNN
from grcnn import GrCNN, GrCNNMatcher
from wordvec import WordEmbedding
from logistic import SoftmaxLayer, LogisticLayer
from utils import floatX
from config import GrCNNConfiger

# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

theano.config.openmp=True
theano.config.on_unused_input='ignore'

class TestGrCNNMatcher(unittest.TestCase):
    '''
    Test the performance of GrCNNMatcher model on matching task.
    '''
    def setUp(self):
        '''
        Load training and test data set, also, load word-embeddings.
        '''
        np.random.seed(42)
        matching_train_filename = '../data/pair_all_sentence_train.txt'
        matching_test_filename = '../data/pair_sentence_test.txt'
        train_pairs_txt, test_pairs_txt = [], []
        # Loading training and test pairs
        start_time = time.time()
        with file(matching_train_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|||')
            for p, q in reader:
                train_pairs_txt.append((p, q))
        with file(matching_test_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|||')
            for p, q in reader:
                test_pairs_txt.append((p, q))
        end_time = time.time()
        logger.debug('Finished loading training and test data set...')
        logger.debug('Time used to load training and test pairs: %f seconds.' % (end_time-start_time))
        embedding_filename = '../data/wiki_embedding.txt'
        word_embedding = WordEmbedding(embedding_filename)
        start_time = time.time()
        # Beginning and trailing token for each sentence
        self.blank_token = word_embedding.wordvec('</s>')
        # Store original text representation
        self.train_pairs_txt, self.test_pairs_txt = train_pairs_txt, test_pairs_txt
        self.train_size = len(self.train_pairs_txt)
        self.test_size = len(self.test_pairs_txt)
        logger.debug('Size of training pairs: %d' % self.train_size)
        logger.debug('Size of test pairs: %d' % self.test_size)
        self.train_pairs_set, self.test_pairs_set = [], []
        # Build word embedding for both training and test data sets
        edim = word_embedding.embedding_dim()
        # Build training data set
        for i, (psent, qsent) in enumerate(self.train_pairs_txt):
            pwords = psent.split()
            pwords = [pword.lower() for pword in pwords]
            pvectors = np.zeros((len(pwords)+2, edim), dtype=floatX)
            pvectors[0, :], pvectors[-1, :] = self.blank_token, self.blank_token
            pvectors[1:-1, :] = np.asrray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

            qwords = qsent.split()
            qwords = [qword.lower() for qword in qwords]
            qvectors = np.zeros((len(qwords)+2, edim), dtype=floatX)
            qvectors[0, :], qvectors[-1, :] = self.blank_token, self.blank_token
            qvectors[1:-1, :] = np.asarray([word_embedding.wordvec(qword) for qword in qwords], dtype=floatX)

            self.train_pairs_set.append((pvectors, qvectors))

        for i, (psent, qsent) in enumerate(self.test_pairs_txt):
            pwords = psent.split()
            pwords = [pword.lower() for pword in pwords]
            pvectors = np.zeros((len(pwords)+2, edim), dtype=floatX)
            pvectors[0, :], pvectors[-1, :] = self.blank_token, self.blank_token
            pvectors[1:-1, :] = np.asrray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

            qwords = qsent.split()
            qwords = [qword.lower() for qword in qwords]
            qvectors = np.zeros((len(qwords)+2, edim), dtype=floatX)
            qvectors[0, :], qvectors[-1, :] = self.blank_token, self.blank_token
            qvectors[1:-1, :] = np.asarray([word_embedding.wordvec(qword) for qword in qwords], dtype=floatX)

            self.test_pairs_set.append((pvectors, qvectors))
        end_time = time.time()
        logger.debug('Training and test data sets building finished...')
        logger.debug('Time used to build training and test data set: %f seconds.' % (end_time-start_time))
        






