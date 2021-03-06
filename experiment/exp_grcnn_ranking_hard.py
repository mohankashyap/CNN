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
import copy
import logging
import traceback
import random
import argparse

from threading import Thread
from multiprocessing import Process, Pool, Queue, Manager
from pprint import pprint

sys.path.append('../source/')

from rnn import BRNN, RNN
from grcnn import GrCNN, GrCNNMatcher, GrCNNMatchScorer
from wordvec import WordEmbedding
from logistic import SoftmaxLayer, LogisticLayer
from utils import floatX
from config import GrCNNConfiger

charas = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
default_name = ''.join([charas[np.random.randint(0, len(charas))] for _ in xrange(5)])
# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

theano.config.openmp=True
theano.config.on_unused_input='ignore'

parser = argparse.ArgumentParser()
device_group = parser.add_mutually_exclusive_group()
device_group.add_argument('-c', '--cpu', type=int, help='Specify the number of cpu kernels to be used.')
device_group.add_argument('-g', '--gpu', action='store_true')
parser.add_argument('-s', '--size', help='The size of each batch used to be trained.',
                    type=int, default=200)
parser.add_argument('-l', '--rate', help='Learning rate of AdaGrad.',
                    type=float, default=1.0)
parser.add_argument('-n', '--name', help='Name used to save the model.',
                    type=str, default=default_name)
parser.add_argument('-m', '--model', help='Model name to use as initializer.',
                    type=str, default='NONE')
parser.add_argument('config', action='store', type=str)

args = parser.parse_args()

np.random.seed(1991)
matching_train_filename = '../data/pair_all_sentence_train.txt'
matching_test_filename = '../data/pair_sentence_test_hard.txt'

train_pairs_txt, test_pairs_txt = [], []
# Loading training and test pairs
start_time = time.time()
with file(matching_train_filename, 'r') as fin:
    for line in fin:
        p, q = line.split('|||')
        train_pairs_txt.append((p, q))
with file(matching_test_filename, 'r') as fin:
    for line in fin:
        p, q, nq = line.split('|||')
        test_pairs_txt.append((p, q, nq))
end_time = time.time()
logger.debug('Finished loading training and test data set...')
logger.debug('Time used to load training and test pairs: %f seconds.' % (end_time-start_time))
embedding_filename = '../data/wiki_embeddings.txt'
word_embedding = WordEmbedding(embedding_filename)
start_time = time.time()
# Beginning and trailing token for each sentence
blank_token = word_embedding.wordvec('</s>')
# Store original text representation
train_size = len(train_pairs_txt)
test_size = len(test_pairs_txt)
logger.debug('Size of training pairs: %d' % train_size)
logger.debug('Size of test pairs: %d' % test_size)
train_pairs_set, test_pairs_set = [], []
# Build word embedding for both training and test data sets
edim = word_embedding.embedding_dim()
# Build training data set
for i, (psent, qsent) in enumerate(train_pairs_txt):
    pwords = psent.split()
    pwords = [pword.lower() for pword in pwords]
    pvectors = np.zeros((len(pwords)+2, edim), dtype=floatX)
    pvectors[0, :], pvectors[-1, :] = blank_token, blank_token
    pvectors[1:-1, :] = np.asarray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

    qwords = qsent.split()
    qwords = [qword.lower() for qword in qwords]
    qvectors = np.zeros((len(qwords)+2, edim), dtype=floatX)
    qvectors[0, :], qvectors[-1, :] = blank_token, blank_token
    qvectors[1:-1, :] = np.asarray([word_embedding.wordvec(qword) for qword in qwords], dtype=floatX)

    train_pairs_set.append((pvectors, qvectors))

for i, (psent, qsent, nqsent) in enumerate(test_pairs_txt):
    pwords = psent.split()
    pwords = [pword.lower() for pword in pwords]
    pvectors = np.zeros((len(pwords)+2, edim), dtype=floatX)
    pvectors[0, :], pvectors[-1, :] = blank_token, blank_token
    pvectors[1:-1, :] = np.asarray([word_embedding.wordvec(pword) for pword in pwords], dtype=floatX)

    qwords = qsent.split()
    qwords = [qword.lower() for qword in qwords]
    qvectors = np.zeros((len(qwords)+2, edim), dtype=floatX)
    qvectors[0, :], qvectors[-1, :] = blank_token, blank_token
    qvectors[1:-1, :] = np.asarray([word_embedding.wordvec(qword) for qword in qwords], dtype=floatX)

    nqwords = nqsent.split()
    nqwords = [nqword.lower() for nqword in nqwords]
    nqvectors = np.zeros((len(nqwords)+2, edim), dtype=floatX)
    nqvectors[0, :], nqvectors[-1, :] = blank_token, blank_token
    nqvectors[1:-1, :] = np.asarray([word_embedding.wordvec(nqword) for nqword in nqwords], dtype=floatX)

    test_pairs_set.append((pvectors, qvectors, nqvectors))
end_time = time.time()
logger.debug('Training and test data sets building finished...')
logger.debug('Time used to build training and test data set: %f seconds.' % (end_time-start_time))
# Set print precision
# np.set_printoptions(threshold=np.nan)
# config_filename = './grCNN_ranker.conf'
config_filename = args.config
start_time = time.time()
configer = GrCNNConfiger(config_filename)
if args.model == 'NONE':
    grcnn = GrCNNMatchScorer(configer, verbose=True)
else:
    grcnn = GrCNNMatchScorer.load(args.model)
end_time = time.time()
logger.debug('Time used to build/load GrCNNMatchRanker: %f seconds.' % (end_time-start_time))
# Define negative/positive sampling ratio
# Begin training
# Using AdaGrad learning algorithm
learn_rate = args.rate
batch_size = args.size
fudge_factor = 1e-6
logger.debug('GrCNNMatchRanker.params: {}'.format(grcnn.params))
hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
# Record the highest training and test accuracy during training process
highest_train_accuracy, highest_test_accuracy = 0.0, 0.0
# Check parameter size
for param in hist_grads:
    logger.debug('Parameter Shape: {}'.format(param.shape))
# Fixing training and test pairs
start_time = time.time()
train_neg_index = range(train_size)
def train_rand(idx):
    nidx = idx
    while nidx == idx: nidx = np.random.randint(0, train_size)
    return nidx
train_neg_index = map(train_rand, train_neg_index)
end_time = time.time()
logger.debug('Time used to generate negative training and test pairs: %f seconds.' % (end_time-start_time))
num_processes = args.cpu

try: 
    # Build multiple workers for parallel processing
    workers = []
    if args.cpu:
        logger.debug('ID of Global GrCNN: {}'.format(id(grcnn)))
        for z in xrange(num_processes):
            start_time = time.time()
            new_worker = GrCNNMatchScorer(configer, verbose=False)
            new_worker.deepcopy(grcnn)
            workers.append(new_worker)
            end_time = time.time()
            logger.debug('Time used to build %d th worker: %f seconds.' % (z, end_time-start_time))
    # Multi-processes for batch learning
    def parallel_process(start_idx, end_idx, worker_id):
        grads, costs, preds = [], 0.0, []
        for j in xrange(start_idx, end_idx):
            sentL, p_sentR = train_pairs_set[j]
            nj = train_neg_index[j]
            n_sentR = train_pairs_set[nj][1]
            r = workers[worker_id].compute_cost_and_gradient(sentL, p_sentR, sentL, n_sentR)
            grad, cost, score_p, score_n = r[:-3], r[-3], r[-2][0], r[-1][0]
            grads.append(grad)
            costs += cost
            preds.append(score_p >= score_n)
        return grads, costs, preds
    # Multi-processes for batch testing
    def parallel_predict(start_idx, end_idx, worker_id):
        costs, preds = 0.0, []
        for j in xrange(start_idx, end_idx):
            sentL, p_sentR, n_sentR = test_pairs_set[j]
            score_p, score_n = workers[worker_id].show_scores(sentL, p_sentR, sentL, n_sentR)
            score_p, score_n = score_p[0], score_n[0]
            if score_p < 1+score_n: costs += 1-score_p+score_n
            preds.append(score_p >= score_n)
        return costs, preds

    start_time = time.time()
    for i in xrange(configer.nepoch):
        logger.debug('-' * 50)
        # Looper over training instances
        total_cost = 0.0
        total_count = 0
        total_predictions = []
        # Compute the number of batches
        num_batch = train_size / batch_size
        logger.debug('Batch size = %d' % batch_size)
        logger.debug('Total number of batches: %d' % num_batch)
        if args.gpu:
            # Start training
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
            hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
            # Using GPU computation
            for j in xrange(train_size):
                if (j+1) % 10000 == 0: logger.debug('%8d @ %4d epoch' % (j+1, i))
                sentL, p_sentR = train_pairs_set[j]
                nj = train_neg_index[j]
                n_sentR = train_pairs_set[nj][1]
                # Call GrCNNMatchRanker
                r = grcnn.compute_cost_and_gradient(sentL, p_sentR, sentL, n_sentR) 
                inst_grads, cost, score_p, score_n = r[:-3], r[-3], r[-2][0], r[-1][0]
                # Accumulate results
                for tot_grad, hist_grad, inst_grad in zip(total_grads, hist_grads, inst_grads):
                    tot_grad += inst_grad
                    hist_grad += np.square(inst_grad)
                total_cost += cost
                total_predictions.append(score_p >= score_n)
                if (j+1) % batch_size == 0 or j == train_size-1:
                    # AdaGrad updating
                    for tot_grad, hist_grad in zip(total_grads, hist_grads):
                        tot_grad /= batch_size
                        tot_grad /= fudge_factor + np.sqrt(hist_grad)
                    # Check total grads
                    grcnn.update_params(total_grads, learn_rate)
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                    hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
        else:
            # Simulate the Map-Reduce strategy to parallelize the training process
            # Using Parallel CPU computation
            # Parallel computing inside each batch
            for j in xrange(num_batch):
                if (j * batch_size) % 10000 == 0: logger.debug('%8d @ %4d epoch' % (j*batch_size, i))
                start_idx = j * batch_size
                step = batch_size / num_processes
                # Creating Process Pool
                pool = Pool(num_processes)
                # Map
                results = []
                for k in xrange(num_processes):
                    results.append(pool.apply_async(parallel_process, args=(start_idx, start_idx+step, k)))
                    start_idx += step
                pool.close()
                pool.join()
                # Accumulate results
                results = [result.get() for result in results]
                total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                # Map-Reduce
                for result in results:
                    grad, cost, pred = result[0], result[1], result[2]
                    for inst_grads in grad:
                        for tot_grad, hist_grad, inst_grad in zip(total_grads, hist_grads, inst_grads):
                            tot_grad += inst_grad
                            hist_grad += np.square(inst_grad)
                    total_cost += cost
                    total_predictions += pred
                # AdaGrad updating
                for tot_grad, hist_grad in zip(total_grads, hist_grads):
                    tot_grad /= batch_size
                    tot_grad /= fudge_factor + np.sqrt(hist_grad)
                # Compute the norm of gradients 
                grcnn.update_params(total_grads, learn_rate)
                # Reduce
                for worker in workers: worker.deepcopy(grcnn)
            # Update all the rests
            if num_batch * batch_size < train_size:
                logger.debug('There are %d training instances need to be processed sequentially.' % (train_size-num_batch*batch_size))
                # Accumulate results
                total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in grcnn.params]
                for j in xrange(num_batch * batch_size, train_size):
                    sentL, p_sentR = train_pairs_set[j]
                    nj = train_neg_index[j]
                    n_sentR = train_pairs_set[nj][1]
                    r = grcnn.compute_cost_and_gradient(sentL, p_sentR, sentL, n_sentR) 
                    inst_grads, cost, score_p, score_n = r[:-3], r[-3], r[-2][0], r[-1][0]
                    for tot_grad, hist_grad, inst_grad in zip(total_grads, hist_grads, inst_grads):
                        tot_grad += inst_grad
                        hist_grad += np.square(inst_grad)
                    total_cost += cost
                    total_predictions.append(score_p >= score_n)
                    # AdaGrad updating
                for tot_grad, hist_grad in zip(total_grads, hist_grads):
                    tot_grad /= train_size - num_batch*batch_size
                    tot_grad /= fudge_factor + np.sqrt(hist_grad)
                # Compute the norm of gradients 
                grcnn.update_params(total_grads, learn_rate)
                # Reduce
                for worker in workers: worker.deepcopy(grcnn)
        # Compute training error
        assert len(total_predictions) == train_size
        total_predictions = np.asarray(total_predictions)
        total_count = np.sum(total_predictions)
        train_accuracy = total_count / float(train_size)
        # logger.debug('-' * 50)
        logger.debug('Total count = {}'.format(total_count))
        # Reporting after each training epoch
        logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, train_accuracy))
        if train_accuracy > highest_train_accuracy: highest_train_accuracy = train_accuracy
        # Testing after each training epoch
        t_num_batch = test_size / batch_size
        test_costs, test_predictions = 0.0, []
        if args.gpu:
            for j in xrange(test_size):
                sentL, p_sentR, n_sentR = test_pairs_set[j]
                score_p, score_n = grcnn.show_scores(sentL, p_sentR, sentL, n_sentR)
                score_p, score_n = score_p[0], score_n[0]
                if score_p < 1+score_n: test_costs += 1-score_p+score_n
                test_predictions.append(score_p >= score_n)
        else:
            for j in xrange(t_num_batch):
                start_idx = j * batch_size
                step = batch_size / num_processes
                # Creating Process Pool
                # Map
                pool = Pool(num_processes)
                results = []
                for k in xrange(num_processes):
                    results.append(pool.apply_async(parallel_predict, args=(start_idx, start_idx+step, k)))
                    start_idx += step
                pool.close()
                pool.join()
                # Accumulate results
                results = [result.get() for result in results]
                # Reduce
                for result in results:
                    test_costs += result[0]
                    test_predictions += result[1]
            if t_num_batch * batch_size < test_size:
                for j in xrange(t_num_batch * batch_size, test_size):
                    sentL, p_sentR, n_sentR = test_pairs_set[j]
                    score_p, score_n = grcnn.show_scores(sentL, p_sentR, sentL, n_sentR)
                    score_p, score_n = score_p[0], score_n[0]
                    if score_p < 1+score_n: test_costs += 1-score_p+score_n
                    test_predictions.append(score_p >= score_n)
        assert len(test_predictions) == test_size
        test_predictions = np.asarray(test_predictions)
        test_accuracy = np.sum(test_predictions) / float(test_size)
        logger.debug('Test accuracy: %f' % test_accuracy)
        logger.debug('Test total cost: %f' % test_costs)
        if test_accuracy > highest_test_accuracy: highest_test_accuracy = test_accuracy
        # Save the model
        logger.debug('Save current model and parameters...')
        GrCNNMatcher.save('GrCNNMatchRanker-{}.pkl'.format(args.name), grcnn)
    end_time = time.time()
    logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
    # Final total test
    start_time = time.time()
    t_num_batch = test_size / batch_size
    logger.debug('Number of test batches: %d' % t_num_batch)
    test_costs, test_predictions = 0.0, []
    for j in xrange(t_num_batch):
        start_idx = j * batch_size
        step = batch_size / num_processes
        # Creating Process Pool
        pool = Pool(num_processes)
        results = []
        for k in xrange(num_processes):
            results.append(pool.apply_async(parallel_predict, args=(start_idx, start_idx+step, k)))
            start_idx += step
        pool.close()
        pool.join()
        # Accumulate results
        results = [result.get() for result in results]
        # Map-Reduce
        for result in results:
            test_costs += result[0]
            test_predictions += result[1]
    if t_num_batch * batch_size < test_size:
        logger.debug('The rest of the test instances are processed sequentially...')
        for j in xrange(t_num_batch * batch_size, test_size):
            sentL, p_sentR, n_sentR = test_pairs_set[j]
            score_p, score_n = grcnn.show_scores(sentL, p_sentR, sentL, n_sentR)
            score_p, score_n = score_p[0], score_n[0]
            if score_p < 1+score_n: test_costs += 1-score_p+score_n
            test_predictions.append(score_p >= score_n)
    assert len(test_predictions) == test_size
    test_predictions = np.asarray(test_predictions)
    logger.debug('Total test predictions = {}'.format(test_predictions))
    test_accuracy = np.sum(test_predictions) / float(test_size)
    end_time = time.time()
    logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
    logger.debug('Test accuracy: %f' % test_accuracy)
    logger.debug('Test total cost: %f' % test_costs)
except:
    logger.debug('!!!Error!!!')
    traceback.print_exc(file=sys.stdout)
    logger.debug('-' * 60)
    if args.cpu and pool != None:
        logger.debug('Quiting all subprocesses...')
        pool.terminate()
finally:
    logger.debug('Highest Training Accuracy: %f' % highest_train_accuracy)
    logger.debug('Highest Test Accuracy: %f' % highest_test_accuracy)
    logger.debug('Saving existing model and parameters...')
    logger.debug('Saving the model: GrCNNMatchRanker-{}.pkl.'.format(args.name))
    GrCNNMatcher.save('GrCNNMatchRanker-{}.pkl'.format(args.name), grcnn)
