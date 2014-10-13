#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-07 15:49:24
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os, sys
import cPickle
import unittest
import time
import logging
import csv
import theano
import theano.tensor as T
import numpy as np
import scipy.io as sio
# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
sys.path.append('../source/')
logger = logging.getLogger(__name__)

from utils import floatX
from grcnn import GrCNN
from wordvec import WordEmbedding
from config import GrCNNConfiger

# theano.config.mode='DebugMode'
theano.config.omp=True
theano.config.on_unused_input='ignore'
np.set_printoptions(threshold=np.nan)

class TestGrCNN(unittest.TestCase):
    def setUp(self):
        '''
        Load training and test texts and labels in sentiment analysis task, preprocessing.
        '''
        np.random.seed(1991)
        senti_train_filename = '../data/sentiment-train.txt'
        senti_test_filename = '../data/sentiment-test.txt'
        senti_train_txt, senti_train_label = [], []
        senti_test_txt, senti_test_label = [], []
        start_time = time.time()
        # Read training data set
        with file(senti_train_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|')
            for txt, label in reader:
                senti_train_txt.append(txt)
                senti_train_label.append(int(label))
        # Read test data set
        with file(senti_test_filename, 'r') as fin:
            reader = csv.reader(fin, delimiter='|')
            for txt, label in reader:
                senti_test_txt.append(txt)
                senti_test_label.append(int(label))
        end_time = time.time()
        logger.debug('Time used to load training and test data set: %f seconds.' % (end_time-start_time))
        embedding_filename = '../data/wiki_embeddings.txt'
        # Load wiki-embeddings
        word_embedding = WordEmbedding(embedding_filename)
        self.token = word_embedding.wordvec('</s>')
        # Store the original text representation
        self.senti_train_txt = senti_train_txt
        self.senti_test_txt = senti_test_txt
        # Word-vector representation
        self.senti_train_label = np.asarray(senti_train_label, dtype=np.int32)
        self.senti_test_label = np.asarray(senti_test_label, dtype=np.int32)
        self.train_size = len(senti_train_txt)
        self.test_size = len(senti_test_txt)
        logger.debug('Training set size: %d' % self.train_size)
        logger.debug('Test set size: %d' % self.test_size)
        assert self.train_size == self.senti_train_label.shape[0]
        assert self.test_size == self.senti_test_label.shape[0]
        # Build the word-embedding matrix
        start_time = time.time()
        self.senti_train_set, self.senti_test_set = [], []
        for sent in senti_train_txt:
            words = sent.split()
            words = [word.lower() for word in words]
            vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
            vectors[0, :] = self.token
            tmp = np.asarray([word_embedding.wordvec(word) for word in words])
            vectors[1:-1, :] = tmp
            vectors[-1, :] = self.token
            self.senti_train_set.append(vectors)
        for sent in senti_test_txt:
            words = sent.split()
            words = [word.lower() for word in words]
            vectors = np.zeros((len(words)+2, word_embedding.embedding_dim()), dtype=floatX)
            vectors[0, :] = self.token
            tmp = np.asarray([word_embedding.wordvec(word) for word in words])
            vectors[1:-1, :] = tmp
            vectors[-1, :] = self.token
            self.senti_test_set.append(vectors)
        end_time = time.time()
        logger.debug('Time used to build training and test word embedding matrix: %f seconds.' % (end_time-start_time))
        self.word_embedding = word_embedding

    def testGrCNNonSentiment(self):
        '''
        Test the ability of GrCNN as an encoder to do sentiment analysis task.
        '''
        conf_filename = './grCNN.conf'
        start_time = time.time()
        configer = GrCNNConfiger(conf_filename)
        grcnn = GrCNN(config=configer, verbose=True)
        end_time = time.time()
        logger.debug('Time used to build GrCNN: %f seconds.' % (end_time-start_time))
        # Training
        start_time = time.time()
        # Loop over epochs
        batch_size = 100
        learning_rate = 0.1
        fudge_factor = 1e-6
        logger.debug('GrCNN.params: {}'.format(grcnn.params))
        history_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
        # Save model parameters
        initial_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
        sio.savemat('grcnn_initial.mat', initial_params)
        # Check
        for param in history_grads:
            logger.debug('Parameter Shape: {}'.format(param.shape))
        for i in xrange(configer.nepoch):
            # Loop over training instances
            total_cost = 0.0
            total_count = 0
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
            # learning_rate = 0.01 / (1.0 + i/10)
            for j in xrange(self.train_size):
                if (j+1) % 1000 == 0:
                    logger.debug('%4d @ %4d epoch' % (j+1, i))
                    current_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
                    sio.savemat('grcnn_{}_{}.mat'.format(i, j), current_params)
                if (j+1) % batch_size == 0 or j == self.train_size-1: 
                    # Adjusted gradient for AdaGrad
                    for grad, hist_grad in zip(total_grads, history_grads):
                        grad /= batch_size
                        grad /= fudge_factor + np.sqrt(hist_grad)
                    grcnn.update_params(total_grads, learning_rate)
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    history_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                results = grcnn.compute_cost_and_gradient(self.senti_train_set[j], [self.senti_train_label[j]])
                grads, cost = results[:-1], results[-1]
                # Accumulate total gradients based on batch size
                for grad, current_grad in zip(total_grads, grads):
                    grad += current_grad
                # Accumulate history gradients based on batch size
                for hist_grad, current_grad in zip(history_grads, grads):
                    hist_grad += np.square(current_grad)
                total_cost += cost
                # Judge whether current instance can be classified correctly or not
                prediction = grcnn.predict(self.senti_train_set[j])[0]
                total_count += prediction == self.senti_train_label[j]
                # # Debugging purpose
                # hidden_rep = grcnn.show_hidden(self.senti_train_set[j], [self.senti_train_label[j]])
                # hidden_compressed_rep = grcnn.show_compressed_hidden(self.senti_train_set[j], [self.senti_train_label[j]])
                # output_rep = grcnn.show_output(self.senti_train_set[j], [self.senti_train_label[j]])
                # logger.debug('=' * 50)
                # logger.debug('Length of gradient vectors: %d' % len(grads))
                # logger.debug('Hidden representation by GrCNN encoder: ')
                # logger.debug(hidden_rep)
                # logger.debug('Hidden compressed with dropout: ')
                # logger.debug(hidden_compressed_rep)
                # logger.debug('Output of the whole architecture: ')
                # logger.debug(output_rep)
                # logger.debug('prediction made by the whole system: ')
                # logger.debug(prediction)
                # logger.debug('Ground truth prediction: ')
                # logger.debug(self.senti_train_label[j])
                # logger.debug('Current cost: ')
                # logger.debug(cost)
                # logger.debug('Gradients: ')
                # logger.debug(grads)
            logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, total_count / float(self.train_size)))
            correct_count = 0
            for j in xrange(self.test_size):
                plabel = grcnn.predict(self.senti_test_set[j])
                if plabel == self.senti_test_label[j]: correct_count += 1
            logger.debug('Test accuracy: %f' % (correct_count / float(self.test_size)))
            # Save model parameters at each iteration
            GrCNN.save('./sentiment.grcnn', grcnn)
        end_time = time.time()
        logger.debug('Time used for training: %f seconds.' % (end_time-start_time))
        start_time = time.time()
        correct_count = 0
        for j in xrange(self.test_size):
            plabel = grcnn.predict(self.senti_test_set[j])
            if plabel == self.senti_test_label[j]: correct_count += 1
        end_time = time.time()
        logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
        logger.debug('Test accuracy: %f' % (correct_count / float(self.test_size)))
        # Save current model onto disk
        GrCNN.save('./sentiment.grcnn', grcnn)


if __name__ == '__main__':
    unittest.main()
