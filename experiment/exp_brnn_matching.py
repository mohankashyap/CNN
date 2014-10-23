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
import argparse

from threading import Thread
from multiprocessing import Process, Pool, Queue
from pprint import pprint

sys.path.append('../source/')

from rnn import BRNN, TBRNN, RNN, BRNNMatcher
from wordvec import WordEmbedding
from logistic import SoftmaxLayer, LogisticLayer
from utils import floatX
from config import RNNConfiger

charas = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
default_name = ''.join([charas[random.randint(0, len(charas)-1)] for _ in xrange(5)])
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
                    type=float, default=1e-1)
parser.add_argument('-n', '--name', help='Name used to save the model.',
                    type=str, default=default_name)
args = parser.parse_args()

np.random.seed(42)
# matching_train_filename = '../data/pair_all_sentence_train.txt'
# matching_test_filename = '../data/pair_sentence_test.txt'
matching_train_filename = '../data/small_pair_train.txt'
matching_test_filename = '../data/small_pair_test.txt'
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
train_pairs_txt, test_pairs_txt = train_pairs_txt, test_pairs_txt
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
# np.set_printoptions(threshold=np.nan)
config_filename = './brnn_matching.conf'
start_time = time.time()
configer = RNNConfiger(config_filename)
brnn = BRNNMatcher(configer, verbose=True)
end_time = time.time()
logger.debug('Time used to build brnn: %f seconds.' % (end_time-start_time))
# Define negative/positive sampling ratio
ratio = 1
logger.debug('Number of positive training pairs: %d' % train_size)
logger.debug('Number of negative training pairs: %d' % train_size * ratio)
# Build training and test index samples
train_index = [(i, i) for i in xrange(train_size)]
train_index += [(np.random.randint(train_size), np.random.randint(train_size)) 
                for _ in xrange(train_size * ratio)]
test_index = [(i, i) for i in xrange(test_size)]
test_index += [(np.random.randint(test_size), np.random.randint(test_size)) 
                for _ in xrange(test_size * ratio)]
# Random shuffle training and test index
random.shuffle(train_index)
random.shuffle(test_index)
# Begin training
# Using AdaGrad learning algorithm
learn_rate = args.rate
batch_size = args.size
fudge_factor = 1e-6
logger.debug('BRNNMatcher.params: {}'.format(brnn.params))
hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
initial_params = {param.name : param.get_value(borrow=True) for param in brnn.params}
sio.savemat('brnn_matcher_initial.mat', initial_params)
# Check parameter size
for param in hist_grads:
    logger.debug('Parameter Shape: {}'.format(param.shape))
# Build training and test labels
train_labels = np.asarray([1 if pidx == qidx else 0 for pidx, qidx in train_index])
test_labels = np.asarray([1 if pidx == qidx else 0 for pidx, qidx in test_index])
# Zip together
start_time = time.time()
tmp_train = [(train_pairs_set[pidx][0], train_pairs_set[qidx][1]) for pidx, qidx in train_index]
tmp_test = [(test_pairs_set[pidx][0], test_pairs_set[qidx][1]) for pidx, qidx in test_index]
train_instances = zip(tmp_train, train_labels)
test_instances = zip(tmp_test, test_labels)
logger.debug('Training instances size, including both positive and negative samples: %d' % len(train_instances))
logger.debug('Test instances size, including both positive and negative samples: %d' % len(test_instances))
end_time = time.time()
logger.debug('Time used to build zipping training and test instances: %f seconds.' % (end_time-start_time))
try: 
    start_time = time.time()
    # Multi-processes for batch learning
    def parallel_process(start_idx, end_idx):
        grads, costs, preds = [], 0.0, []
        for (sentL, sentR), label in train_instances[start_idx: end_idx]:
            r = brnn.compute_cost_and_gradient(sentL, sentR, [label])
            grad, cost, pred = r[:-2], r[-2], r[-1]
            if len(grads) == 0:
                grads, costs, preds = grad, cost, [pred[0]]
            else:
                for gt, g in zip(grads, grad):
                    gt += g
                costs += cost
                preds.append(pred[0])
        # Each element of results is a three-element tuple, where the first element
        # accumulates the gradients, the second element accumulate the cost and the 
        # third element store the predictions
        return grads, costs, preds

    for i in xrange(configer.nepoch):
        # Looper over training instances
        total_cost = 0.0
        total_count = 0
        total_predictions = []
        # Compute the number of batches
        num_batch = len(train_instances) / batch_size
        logger.debug('Batch size = %d' % batch_size)
        logger.debug('Total number of batches: %d' % num_batch)
        if args.gpu:
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
            hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
            # Using GPU computation
            for j in xrange(len(train_instances)):
                if (j+1) % 10000 == 0: logger.debug('%8d @ %4d epoch' % (j+1, i))
                (sentL, sentR), label = train_instances[j]
                r = brnn.compute_cost_and_gradient(sentL, sentR, [label]) 
                grad, cost, pred = r[:-2], r[-2], r[-1]
                # Accumulate results
                for gt, g in zip(total_grads, grad):
                    gt += g
                for gt, g in zip(hist_grads, grad):
                    gt += np.square(g)
                total_cost += cost
                total_predictions.append(pred[0])
                if (j+1) % batch_size == 0 or j == len(train_instances)-1:
                    # AdaGrad updating
                    for grad, hist_grad in zip(total_grads, hist_grads):
                        grad /= batch_size
                        grad /= fudge_factor + np.sqrt(hist_grad)
                    brnn.update_params(total_grads, learn_rate)
                    total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
                    hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
        else:
            num_processes = args.cpu
            # Using Parallel CPU computation
            # Parallel computing inside each batch
            for j in xrange(num_batch):
                if (j * batch_size) % 10000 == 0: logger.debug('%8d @ %4d epoch' % (j*batch_size, i))
                start_idx = j * batch_size
                step = batch_size / num_processes
                # Creating Process Pool
                pool = Pool(num_processes)
                results = []
                for k in xrange(num_processes):
                    results.append(pool.apply_async(parallel_process, args=(start_idx, start_idx+step)))
                    start_idx += step
                pool.close()
                pool.join()
                # Accumulate results
                results = [result.get() for result in results]
                total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
                hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
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
                brnn.update_params(total_grads, learn_rate)
            if num_batch * batch_size < len(train_instances):
                # Accumulate results
                total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
                hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in brnn.params]
                # Update all the rests
                for j in xrange(num_batch * batch_size, len(train_instances)):
                    (sentL, sentR), label = train_instances[j]
                    r = brnn.compute_cost_and_gradient(sentL, sentR, [label]) 
                    grad, cost, pred = r[:-2], r[-2], r[-1]
                    for gt, g in zip(total_grads, grad):
                        gt += g
                    for gt, g in zip(hist_grads, grad):
                        gt += np.square(g)
                    total_cost += cost
                    total_predictions.append(pred[0])
                # AdaGrad updating
                for grad, hist_grad in zip(total_grads, hist_grads):
                    grad /= len(train_index) - num_batch*batch_size
                    grad /= fudge_factor + np.sqrt(hist_grad)
                brnn.update_params(total_grads, learn_rate)
        # Compute training error
        total_predictions = np.asarray(total_predictions)
        train_labels = np.asarray(train_labels)
        total_count = np.sum(total_predictions == train_labels)
        # Reporting after each training epoch
        logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, total_count / float(len(train_instances))))
        correct_count = 0
        for j in xrange(len(test_instances)):
            (sentL, sentR), label = test_instances[j]
            plabel = brnn.predict(sentL, sentR)[0]
            if label == plabel: correct_count += 1
        logger.debug('Test accuracy: %f' % (correct_count / float(len(test_index))))
        # Save the model
        logger.debug('Save current model...')
        BRNNMatcher.save('brnnMatcher-{}.pkl'.format(args.name), brnn)
    end_time = time.time()
    logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
    # Final total test
    start_time = time.time()
    correct_count = 0
    for j in xrange(len(test_instances)):
        (sentL, sentR), label = test_instances[j]
        plabel = brnn.predict(sentL, sentR)[0]
        if label == plabel: correct_count += 1
    end_time = time.time()
    logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
    logger.debug('Test accuracy: %f' % (correct_count / float(len(test_index))))
except:
    logger.debug('!!!Error!!!')
    logger.debug('-' * 60)
    if args.cpu and pool != None:
        logger.debug('Quiting all subprocesses...')
        pool.terminate()
    traceback.print_exc(file=sys.stdout)
finally:            
    logger.debug('Saving the model: brnnMatcher-{}.pkl.'.format(args.name))
    BRNNMatcher.save('brnnMatcher-{}.pkl'.format(args.name), brnn)

