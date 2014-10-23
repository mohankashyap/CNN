#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-09-04 15:42:49
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
sys.path.append('../source/')
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

from rnn import BRNN, TBRNN, RNN
from config import RNNConfiger
from wordvec import WordEmbedding
from logistic import SoftmaxLayer
from utils import floatX

theano.config.openmp=True
theano.config.on_unused_input='ignore'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)


class TestRNNSP(unittest.TestCase):
    '''
    Test the performance of BRNN model on subjective-passive classification task.
    '''
    def setUp(self):
        '''
        Load training and test data set, also, loading word-embeddings.
        '''
        np.random.seed(1991)
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
        # Read test data set
        with file(sp_test_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|')
            for txt, label in reader:
                sp_test_txt.append(txt)
                sp_test_label.append(label)
        end_time = time.time()
        logger.debug('Finished loading training and test data sets...')
        logger.debug('Time used for loading: %f seconds.' % (end_time-start_time))
        embedding_filename = '../data/wiki_embeddings.txt'
        word_embedding = WordEmbedding(embedding_filename)
        start_time = time.time()
        # Starting and Ending token for each sentence
        self.blank_token = word_embedding.wordvec('</s>')
        # Store original text representation
        self.sp_train_txt = sp_train_txt
        self.sp_test_txt = sp_test_txt
        # Word-vector representation
        self.sp_train_label = np.asarray(sp_train_label, dtype=np.int32)
        self.sp_test_label = np.asarray(sp_test_label, dtype=np.int32)
        train_size = len(sp_train_txt)
        test_size = len(sp_test_txt)
        # Check size
        assert train_size == self.sp_train_label.shape[0]
        assert test_size == self.sp_test_label.shape[0]
        logger.debug('Training size: %d' % train_size)
        logger.debug('Test size: %d' % test_size)
        # Sequential modeling for each sentence
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
        assert sp_train_len == [seq.shape[0] for seq in self.sp_train_set]
        assert sp_test_len == [seq.shape[0] for seq in self.sp_test_set]
        end_time = time.time()
        logger.debug('Time used to build initial training and test matrix: %f seconds.' % (end_time-start_time))
        # Store data
        self.train_size = train_size
        self.test_size = test_size
        self.word_embedding = word_embedding
        logger.debug('Max sentence length in training set: %d' % max(sp_train_len))
        logger.debug('Max sentence length in test set: %d' % max(sp_test_len))

    def testTBRNN(self):
        # Set print precision
        np.set_printoptions(threshold=np.nan)

        config_filename = './sp_brnn.conf'
        start_time = time.time()
        configer = RNNConfiger(config_filename)
        brnn = BRNN(configer, verbose=True)
        end_time = time.time()
        logger.debug('Time used to build TBRNN: %f seconds.' % (end_time-start_time))
        # Training
        logger.debug('positive labels: %d' % np.sum(self.sp_train_label))
        logger.debug('negative labels: %d' % (self.sp_train_label.shape[0]-np.sum(self.sp_train_label)))
        start_time = time.time()
        ## AdaGrad learning algorithm instead of the stochastic gradient descent algorithm
        # history_grads = np.zeros(brnn.num_params)
        n_epoch = 200
        learn_rate = 1e-1
        batch_size = 100
        fudge_factor = 1e-6
        history_grads = np.zeros(brnn.num_params, dtype=floatX)
        logger.debug('Number of parameters in the model: {}'.format(brnn.num_params))
        for i in xrange(n_epoch):
            tot_count = 0
            tot_error = 0.0
            tot_grads = np.zeros(brnn.num_params, dtype=floatX)
            for j, (train_seq, train_label) in enumerate(zip(self.sp_train_set, self.sp_train_label)):
                if (j+1) % 1000 == 0:
                    logger.debug('%4d @ %4d epoch' % (j+1, i))
                cost, current_grads = brnn.compute_cost_and_gradient(train_seq, [train_label])
                # Accumulate current gradients
                tot_grads += current_grads
                tot_error += cost
                # Accumulate historical gradients
                history_grads += np.square(current_grads)
                # predict current training label
                prediction = brnn.predict(train_seq)[0]
                tot_count += prediction == train_label
                # Batch updating with AdaGrad
                if (j+1) % batch_size == 0 or j == self.train_size-1:
                    # Adjust gradient for AdaGrad
                    tot_grads /= batch_size
                    tot_grads /= fudge_factor + np.sqrt(history_grads)
                    brnn.update_params(tot_grads, learn_rate)
                    tot_grads = np.zeros(brnn.num_params, dtype=floatX)
                    history_grads = np.zeros(brnn.num_params, dtype=floatX)
            logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, tot_error, tot_count / float(self.train_size)))
            # Testing
            tot_count = 0
            for test_seq, test_label in zip(self.sp_test_set, self.sp_test_label):
                prediction = brnn.predict(test_seq)[0]
                tot_count += test_label == prediction
            logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
        end_time = time.time()
        logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
        # Testing
        tot_count = 0
        for test_seq, test_label in zip(self.sp_test_set, self.sp_test_label):
            prediction = brnn.predict(test_seq)[0]
            tot_count += test_label == prediction
        logger.debug('Test accuracy: %f' % (tot_count / float(self.test_size)))
        logger.debug('Percentage of positive in Test data: %f' % (np.sum(self.sp_test_label==1) / float(self.test_size)))
        logger.debug('Percentage of negative in Test data: %f' % (np.sum(self.sp_test_label==0) / float(self.test_size)))
        # Re-testing on training set
        tot_count = 0
        for train_seq, train_label in zip(self.sp_train_set, self.sp_train_label):
            prediction = brnn.predict(train_seq)[0]
            tot_count += train_label == prediction
        logger.debug('Training accuracy re-testing: %f' % (tot_count / float(self.train_size)))
        # Save TBRNN
        BRNN.save('sp.brnn.pkl', brnn)
        logger.debug('Model successfully saved...')


if __name__ == '__main__':
    unittest.main()

