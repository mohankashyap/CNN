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
from config import GrCNNConfiger
# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

theano.config.openmp=True
theano.config.on_unused_input='ignore'

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
        self.sp_test_txt = sp_test_txt
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
        logger.debug('Time used to build initial training and test matrix: %f seconds' % (end_time-start_time))
        # Store metadata
        self.train_size = train_size
        self.test_size = test_size
        self.word_embedding = word_embedding
        logger.debug('Sentence of maximum length in training set: %d' % max(sp_train_len))
        logger.debug('Sentence of maximum length in test set: %d' % max(sp_test_len))

    def testGrCNN(self):
        # Set print precision
        np.set_printoptions(threshold=np.nan)
        config_filename = './grCNN.conf'
        start_time = time.time()
        configer = GrCNNConfiger(config_filename)
        grcnn = GrCNN(configer, verbose=True) 
        end_time = time.time()
        logger.debug('Time used to build GrCNN: %f seconds.' % (end_time-start_time))
        logger.debug('Positive labels: %d' % np.sum(self.sp_train_label))
        logger.debug('Negative labels: %d' % (self.sp_train_label.shape[0]-np.sum(self.sp_train_label)))
        # Training using GrCNN
        start_time = time.time()
        learn_rate = 1e-1
        batch_size = 100
        fudge_factor = 1e-6
        logger.debug('GrCNN.params: {}'.format(grcnn.params))
        history_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
        initial_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
        sio.savemat('grcnn_initial.mat', initial_params)
        # Check parameter size
        for param in history_grads:
            logger.debug('Parameter Shape: {}'.format(param.shape))
        for i in xrange(configer.nepoch):
            # Loop over training instances
            total_cost = 0.0
            total_count = 0
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
            # AdaGrad
            for j in xrange(self.train_size):
                if (j+1) % 1000 == 0:
                    logger.debug('%4d @ %4d epoch' % (j+1, i))
                    current_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
                    sio.savemat('grcnn_{}_{}.mat'.format(i, j+1), current_params)
                results = grcnn.compute_cost_and_gradient(self.sp_train_set[j], [self.sp_train_label[j]])
                grads, cost = results[:-1], results[-1]
                # Accumulate total gradients based on batch size
                for grad, current_grad in zip(total_grads, grads):
                    grad += current_grad
                # Accumulate history gradients based on batch size
                for hist_grad, current_grad in zip(history_grads, grads):
                    hist_grad += np.square(current_grad)
                # Judge whether current instance can be classified correctly or not  
                prediction = grcnn.predict(self.sp_train_set[j])[0]
                total_count += prediction == self.sp_train_label[j]
                total_cost += cost
                if (j+1) % batch_size == 0 or j == self.train_size-1:
                    # Adjusted gradient for AdaGrad
                    for grad, hist_grad in zip(total_grads, history_grads):
                        grad /= batch_size
                        grad /= fudge_factor + np.sqrt(hist_grad)
                    grcnn.update_params(total_grads, learn_rate)
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    history_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
            logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, total_count / float(self.train_size)))
            correct_count = 0
            for j in xrange(self.test_size):
                plabel = grcnn.predict(self.sp_test_set[j])
                if plabel == self.sp_test_label[j]: correct_count += 1
            logger.debug('Test accuracy: %f' % (correct_count / float(self.test_size)))
        end_time = time.time()
        logger.debug('Time used for training: %f minutes.' % ((end_time-start_time) / 60))
        # Final total test
        start_time = time.time()
        correct_count = 0
        for j in xrange(self.test_size):
            plabel = grcnn.predict(self.sp_test_set[j])
            if plabel == self.sp_test_label[j]: correct_count += 1
        end_time = time.time()
        logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
        logger.debug('Test accuracy: %f' % (correct_count / float(self.test_size)))
        # Save current model
        GrCNN.save('./subjective-passive.grcnn', grcnn)



if __name__ == '__main__':
    unittest.main()














