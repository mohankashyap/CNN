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
import traceback
import random

from threading import Thread
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
            for line in fin:
                p, q = line.split('|||')
                train_pairs_txt.append((p, q))
        with file(matching_test_filename, 'r') as fin:
            for line in fin:
                p, q = line.split('|||')
                test_pairs_txt.append((p, q))
        end_time = time.time()
        logger.debug('Finished loading training and test data set...')
        logger.debug('Time used to load training and test pairs: %f seconds.' % (end_time-start_time))
        embedding_filename = '../data/wiki_embeddings.txt'
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
            pvectors[1:-1, :] = np.asarray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

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
            pvectors[1:-1, :] = np.asarray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

            qwords = qsent.split()
            qwords = [qword.lower() for qword in qwords]
            qvectors = np.zeros((len(qwords)+2, edim), dtype=floatX)
            qvectors[0, :], qvectors[-1, :] = self.blank_token, self.blank_token
            qvectors[1:-1, :] = np.asarray([word_embedding.wordvec(qword) for qword in qwords], dtype=floatX)

            self.test_pairs_set.append((pvectors, qvectors))
        end_time = time.time()
        logger.debug('Training and test data sets building finished...')
        logger.debug('Time used to build training and test data set: %f seconds.' % (end_time-start_time))
        self.word_embedding = word_embedding

    def testGrCNNMatching(self):
        # Set print precision
        np.set_printoptions(threshold=np.nan)
        config_filename = './grCNN.conf'
        start_time = time.time()
        configer = GrCNNConfiger(config_filename)
        grcnn = GrCNNMatcher(configer, verbose=True)
        end_time = time.time()
        logger.debug('Time used to build GrCNN: %f seconds.' % (end_time-start_time))
        # Define negative/positive sampling ratio
        ratio = 1
        logger.debug('Number of positive training pairs: %d' % self.train_size)
        logger.debug('Number of negative training pairs: %d' % self.train_size * ratio)
        # Build training and test index samples
        train_index = [(i, i) for i in xrange(self.train_size)]
        train_index += [(np.random.randint(self.train_size), np.random.randint(self.train_size)) 
                        for _ in xrange(self.train_size * ratio)]
        test_index = [(i, i) for i in xrange(self.test_size)]
        test_index += [(np.random.randint(self.test_size), np.random.randint(self.test_size)) 
                        for _ in xrange(self.test_size * ratio)]
        # Random shuffle training and test index
        random.shuffle(train_index)
        random.shuffle(test_index)
        # Begin training
        start_time = time.time()
        # Using AdaGrad learning algorithm
        learn_rate = 1e-1
        batch_size = 100
        fudge_factor = 1e-6
        logger.debug('GrCNNMatcher.params: {}'.format(grcnn.params))
        hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
        initial_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
        sio.savemat('grcnn_matcher_initial.mat', initial_params)
        # Check parameter size
        for param in hist_grads:
            logger.debug('Parameter Shape: {}'.format(param.shape))
        # Build training and test labels
        train_labels = np.asarray([1 if pidx == qidx else 0 for pidx, qidx in train_index])
        test_labels = np.asarray([1 if pidx == qidx else 0 for pidx, qidx in test_index])
        try: 
            # Multi-threading process for each batch
            num_threads = 20
            threads = [None] * num_threads
            results = [None] * num_threads
            # Local processing function for each thread
            def thread_process(idx, start_idx, end_idx):
                '''
                @idx: Int. Store result in the idx th cell.
                @start_idx: Int. Starting index of current processing, inclusive.
                @end_idx: Int. Ending index of current processing, exclusive.
                '''
                grads, costs, preds = [], 0.0, []
                for k in xrange(start_idx, end_idx):
                    pidx, qidx = self.train_index[k][0], self.train_index[k][1]
                    label = 1 if pidx == qidx else 0
                    r = grcnn.compute_cost_and_gradient(self.train_pairs_set[pidx][0], 
                                                        self.train_pairs_set[qidx][1],
                                                        [label])
                    grad, cost, pred = r[:-2], r[-2], r[-1]
                    if len(grads) == 0:
                        grads, costs, preds = grad, cost, pred
                    else:
                        for gt, g in zip(grads, grad):
                            gt += g
                        costs += cost
                        preds += pred
                # Each element of results is a three-element tuple, where the first element
                # accumulates the gradients, the second element accumulate the cost and the 
                # third element store the predictions
                results[idx] = (grads, costs, preds)

            for i in xrange(configer.nepoch):
                # Looper over training instances
                total_cost = 0.0
                total_count = 0
                total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                total_predictions = []
                # Compute the number of batches
                num_batch = len(train_index) / batch_size
                # Parallel computing inside each batch
                for j in xrange(num_batch):
                    if j * batch_size == 10000: logger.debug('%8d @ %4d epoch' % (j, i))
                    start_idx = j * batch_size
                    step = batch_size / num_threads
                    for k in xrange(num_threads):
                        threads[k] = Thread(target=thread_process, args=(k, start_idx, start_idx+step))
                        threads[k].start()
                        start_idx += step
                    # Threads working
                    for k in xrange(num_threads):
                        threads[k].join()
                    # Accumulate results
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    for result in results:
                        grad, cost, pred = result[0], result[1], result[2]
                        for gt, g in zip(total_grads, grad):
                            gt += g
                        for gt, g in zip(hist_grads, grad):
                            gt += np.square(g)
                        total_cost += cost
                        total_predictions += pred
                    # AdaGrad updating
                    for grad, hist_grad in zip(total_grads, hist_grads):
                        grad /= batch_size
                        grad /= fudge_factor + np.sqrt(hist_grad)
                    grcnn.update_params(total_grads, learn_rate)
                # Update all the rests
                for j in xrange(num_batch * batch_size, len(train_index)):
                    pidx, qidx = train_index[j][0], train_index[j][1]
                    label = 1 if pidx == qidx else 0
                    r = grcnn.compute_cost_and_gradient(self.train_pairs_set[pidx][0], 
                                                        self.train_paris_set[qidx][1],
                                                        [label])
                    grad, cost, pred = r[:-2], r[-2], r[-1]
                    # Accumulate results
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    for result in results:
                        grad, cost, pred = result[0], result[1], result[2]
                        for gt, g in zip(total_grads, grad):
                            gt += g
                        for gt, g in zip(hist_grads, grad):
                            gt += np.square(g)
                        total_cost += cost
                        total_predictions += pred
                    # AdaGrad updating
                    for grad, hist_grad in zip(total_grads, hist_grads):
                        grad /= len(train_index) - num_batch*batch_size
                        grad /= fudge_factor + np.sqrt(hist_grad)
                    grcnn.update_params(total_grads, learn_rate)
                # Compute training error
                total_predictions = np.asarray(total_predictions)
                total_count = np.sum(total_predictions == train_labels)
                # # AdaGrad
                # for j, (pidx, qidx) in enumerate(train_index):
                #     if (j+1) % 10000 == 0: logger.debug('%8d @ %4d epoch' % (j+1, i))
                #     label = 1 if pidx == qidx else 0
                #     results = grcnn.compute_cost_and_gradient(self.train_pairs_set[pidx][0], 
                #                                               self.train_pairs_set[qidx][1], 
                #                                               [label])
                #     grads, cost = results[:-1], results[-1]
                #     # Accumulate total gradients based on batch size
                #     for tot_grad, grad in zip(total_grads, grads):
                #         tot_grad += grad
                #     # Accumulate historical gradients based on batch size
                #     for hist_grad, grad in zip(hist_grads, grads):
                #         hist_grad += np.square(grad)
                #     # Judge whether current instance can be classified correctly or not
                #     prediction = grcnn.predict(self.train_pairs_set[pidx][0], 
                #                                self.train_pairs_set[qidx][1])[0]
                #     total_count += prediction == label
                #     total_cost += cost
                #     # Update parameters based on batch mode
                #     if (j+1) % batch_size == 0 or j == len(train_index)-1:
                #         # AdaGrad 
                #         for grad, hist_grad in zip(total_grads, hist_grads):
                #             grad /= batch_size
                #             grad /= fudge_factor + np.sqrt(hist_grad)
                #         grcnn.update_params(total_grads, learn_rate)
                #         total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                #         hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, total_count / float(len(train_index))))
                correct_count = 0
                for j, (pidx, qidx) in enumerate(test_index):
                    label = 1 if pidx == qidx else 0
                    plabel = grcnn.predict(self.test_pairs_set[pidx][0], self.test_pairs_set[qidx][1])[0]
                    if label == plabel: correct_count += 1
                logger.debug('Test accuracy: %f' % (correct_count / float(len(test_index))))
            end_time = time.time()
            logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
            # Final total test
            start_time = time.time()
            correct_count = 0
            for pidx, qidx in test_index:
                label = 1 if pidx == qidx else 0
                plabel = grcnn.predict(self.test_pairs_set[pidx][0], self.test_pairs_set[qidx][1])[0]
                if label == plabel: correct_count += 1
            end_time = time.time()
            logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
            logger.debug('Test accuracy: %f' % (correct_count / float(len(test_index))))
        except:
            logger.debug('!!!Error!!!')
            logger.debug('-' * 60)
            traceback.print_exc(file=sys.stdout)
            logger.debug('-' * 60)
        finally:            
            final_params = {param.name : param.get_value(borrow=True) for param in grcnn.params}
            sio.savemat('grcnn_matcher_final.mat', initial_params)
            GrCNNMatcher.save('GrCNNMatcher.pkl', grcnn)

if __name__ == '__main__':
    unittest.main()


