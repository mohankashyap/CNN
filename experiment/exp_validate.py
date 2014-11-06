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
from utils import floatX
from config import GrCNNConfiger, RNNConfiger

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
parser.add_argument('-s', '--seed', help='Random seed.',
                    type=int, default=42)
parser.add_argument('-d', '--size', help='Batch size.')
parser.add_argument('-r', '--ratio', help='Ratio of the whole test data to be tested.', 
                    type=float, default=1.0)

args = parser.parse_args()

np.random.seed(args.seed)
matching_train_filename = '../data/pair_all_sentence_train.txt'
matching_test_filename = '../data/pair_sentence_test.txt'
model_filename = './GrCNNMatchRanker-RANK-DROPOUT.pkl'

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

for i, (psent, qsent) in enumerate(test_pairs_txt):
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

    test_pairs_set.append((pvectors, qvectors))
end_time = time.time()
logger.debug('Training and test data sets building finished...')
logger.debug('Time used to build training and test data set: %f seconds.' % (end_time-start_time))
# Set print precision
start_time = time.time()
configer = GrCNNConfiger('./grCNN_ranker.conf')
grcnn = GrCNNMatchScorer.load(model_filename)
end_time = time.time()
logger.debug('Time used to build/load GrCNNMatchRanker: %f seconds.' % (end_time-start_time))
# Output Model size
for param in grcnn.params:
    logger.debug('Parameter {}: {}'.format(param.name, param.shape))
# Define negative/positive sampling ratio
# Check parameter size
logger.debug('Number of parameters in this model: {}'.format(grcnn.num_params))
# Fixing training and test pairs
start_time = time.time()
train_neg_index = range(train_size)
test_neg_index = range(test_size)
def train_rand(idx):
    nidx = idx
    while nidx == idx: nidx = np.random.randint(0, train_size)
    return nidx
def test_rand(idx):
    nidx = idx
    while nidx == idx: nidx = np.random.randint(0, test_size)
    return nidx
train_neg_index = map(train_rand, train_neg_index)
test_neg_index = map(test_rand, test_neg_index)
end_time = time.time()
logger.debug('Time used to generate negative training and test pairs: %f seconds.' % (end_time-start_time))
num_processes = args.cpu

test_size = int(test_size * args.ratio)

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
    # Multi-processes for batch testing
    def parallel_predict(start_idx, end_idx, worker_id):
        costs, preds = 0.0, []
        for j in xrange(start_idx, end_idx):
            sentL, p_sentR = test_pairs_set[j]
            nj = test_neg_index[j]
            n_sentR = test_pairs_set[nj][1]
            score_p, score_n = workers[worker_id].show_scores(sentL, p_sentR, sentL, n_sentR)
            score_p, score_n = score_p[0], score_n[0]
            if score_p < 1+score_n: costs += 1-score_p+score_n
            preds.append(score_p >= score_n)
        return costs, preds
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
            sentL, p_sentR = test_pairs_set[j]
            nj = test_neg_index[j]
            n_sentR = test_pairs_set[nj][1]
            score_p, score_n = grcnn.show_scores(sentL, p_sentR, sentL, n_sentR)
            score_p, score_n = score_p[0], score_n[0]
            if score_p < 1+score_n: test_costs += 1-score_p+score_n
            test_predictions.append(score_p >= score_n)
    assert len(test_predictions) == test_size
    test_predictions = np.asarray(test_predictions)
    logger.debug('Number of test cases: %d' % test_size)
    logger.debug('Total test costs: %f' % test_costs)
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
    logger.debug('Safely quit the validation program...')
