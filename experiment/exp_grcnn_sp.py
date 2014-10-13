#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-13 11:28:22
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
from grcnn import GrCNN
from wordvec import WordEmbedding
from logistic import SoftmaxLayer
from utils import floatX
# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

theano.config.openmp=True


class TestGrCNNSP(unittest.TestCase):
    '''
    Test the performance of GrCNN model on subjective-passive classification task.
    '''
    def setUp(self):
        '''
        Load training and test data set, also, loading word-embeddings.
        '''
        np.random.seed(42)
        sp_train_filename = '../data/refined_train_sp.txt'
        sp_test_filename = '../data/refined_test_sp.txt'    
        sp_train_txt, sp_train_label = [], []
        sp_test_txt, sp_test_label = [], []
        start_time = time.time()
        # Read training data set
        with file(sp_train_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|')
            for txt, label in reader:
                sp_train_txt.append(txt)
                sp_train_label.append(int(label))
        with file(sp_test_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|')
            for txt, label in reader:
                sp_test_txt.append(txt)
                sp_test_label.append(int(label))
        end_time = time.time()
        logger.debug('Finished loading training and test data set...')
        logger.debug('Time used for loading: %f seconds.' % (end_time-start_time))
        embedding_filename = '../data/wiki_embeddings.txt'
        word_embedding = WordEmbedding(embedding_filename)
        start_time = time.time()
        # Beginning and trailing token for each sentence
        self.blank_token = word_embedding.wordvec('</s>')
        # Store original text representation
        self.sp_train_txt = sp_train_txt
        self.sp_test_txt = sp_text_txt
        # Store original label
        self.sp_train_label = np.asarray(sp_train_label, dtype=np.int32)
        self.sp_test_label = np.asarray(sp_test_label, dtype=np.int32)
        train_size = len(sp_train_txt)
        test_size = len(sp_test_txt)
        # Check size
        assert train_size == self.sp_train_label.shape[0]
        assert test_size == self.sp_test_label.shape[0]
        # Output the information
        logger.debug('Training size: %d' % train_size)
        logger.debug('Test size: %d' % test_size)
        # Word-vector representation
        self.sp_train_set, self.sp_test_set = [], []
        sp_train_len, sp_test_len = [], []
        # Embedding for training set
        for i, sent in enumerate(sp_train_txt):
            words = sent.split()
            words = [word.lower() for word in words]
            vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
            vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
            sp_train_len.append(len(words)+2)
            self.sp_train_set.append(vectors)
        # Embedding for test set
        for i, sent in enumerate(sp_test_txt):
            words = sent.split()
            words = [word.lower() for word in words]
            vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
            vectors[1:-1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
            sp_test_len.append(len(words)+2)
            self.sp_test_set.append(vectors)
        # Check word-length
        assert sp_train_len == [seq.shape[0] for seq in self.sp_train_set]
        assert sp_test_len == [seq.shape[0] for seq in self.sp_test_set]
        end_time = time.time()
        logger.debug('Time used to build initial training and test matrix: %f seconds'. % (end_time-start_time))
        # Store metadata
        self.train_size = train_size
        self.test_size = test_size
        self.word_embedding = word_embedding
        logger.debug('Sentence of maximum length in training set: %d' % max(sp_train_len))
        logger.debug('Sentence of maximum length in test set: %d' % max(sp_test_len))

    






















