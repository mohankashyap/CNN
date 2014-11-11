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

from pprint import pprint

sys.path.append('../source/')

from wordvec import WordEmbedding
from logistic import SoftmaxLayer, LogisticLayer
from mlp import HiddenLayer
from score import ScoreLayer
from activations import Activation
from utils import floatX

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
parser.add_argument('-d', '--hidden', help='The size of the hidden layer in MLP.', 
                    type=int, default=50)
parser.add_argument('-s', '--size', help='The size of each batch used to be trained.',
                    type=int, default=200)
parser.add_argument('-l', '--rate', help='Learning rate of AdaGrad.',
                    type=float, default=1.0)
parser.add_argument('-n', '--name', help='Name used to save the model.',
                    type=str, default=default_name)
parser.add_argument('-p', '--dropout', help='Dropout parameter.', 
                    type=float, default=0.0)
parser.add_argument('-r', '--seed', help='Random seed.',
                    type=int, default=42)
parser.add_argument('config', action='store', type=str)
args = parser.parse_args()

np.random.seed(args.seed)
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
start_time = time.time()
# Ranking task using Multilayer Perceptron Model
class MLPRanker(object):
    def __init__(self, verbose=True):
        if verbose: logger.debug('Build Multilayer Perceptron Ranking model...')
        # Positive input setting
        self.inputPL = T.matrix(name='inputPL', dtype=floatX)
        self.inputPR = T.matrix(name='inputPR', dtype=floatX)
        # Negative input setting
        self.inputNL = T.matrix(name='inputNL', dtype=floatX)
        self.inputNR = T.matrix(name='inputNR', dtype=floatX)
        # Standard input setting
        self.inputL = T.matrix(name='inputL', dtype=floatX)
        self.inputR = T.matrix(name='inputR', dtype=floatX)
        # Build activation function
        self.act = Activation('tanh')
        # Connect input matrices
        self.inputP = T.concatenate([self.inputPL, self.inputPR], axis=1)
        self.inputN = T.concatenate([self.inputNL, self.inputNR], axis=1)
        self.input = T.concatenate([self.inputL, self.inputR], axis=1)
        # Build hidden layer
        self.hidden_layer = HiddenLayer(self.input, (2*edim, args.hidden), act=self.act)
        self.hidden = self.hidden_layer.output
        self.hiddenP = self.hidden_layer.encode(self.inputP)
        self.hiddenN = self.hidden_layer.encode(self.inputN)
        # Dropout parameter
        #srng = T.shared_randomstreams.RandomStreams(args.seed)
        #mask = srng.binomial(n=1, p=1-args.dropout, size=self.hidden.shape)
        #maskP = srng.binomial(n=1, p=1-args.dropout, size=self.hiddenP.shape)
        #maskN = srng.binomial(n=1, p=1-args.dropout, size=self.hiddenN.shape)
        #self.hidden *= T.cast(mask, floatX)
        #self.hiddenP *= T.cast(maskP, floatX)
        #self.hiddenN *= T.cast(maskN, floatX)
        # Build linear output layer
        self.score_layer = ScoreLayer(self.hidden, args.hidden)
        self.output = self.score_layer.output
        self.scoreP = self.score_layer.encode(self.hiddenP)
        self.scoreN = self.score_layer.encode(self.hiddenN)
        # Stack all the parameters
        self.params = []
        self.params += self.hidden_layer.params
        self.params += self.score_layer.params
        # Build cost function
        self.cost = T.mean(T.maximum(T.zeros_like(self.scoreP), 1.0-self.scoreP+self.scoreN))
        # Construct the gradient of the cost function with respect to the model parameters
        self.gradparams = T.grad(self.cost, self.params)
        # Count the total number of parameters in this model
        self.num_params = edim * args.hidden + args.hidden + args.hidden + 1
        # Build class method
        self.score = theano.function(inputs=[self.inputL, self.inputR], outputs=self.output)
        self.compute_cost_and_gradient = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR],
                                                         outputs=self.gradparams+[self.cost, self.scoreP, self.scoreN])
        self.show_scores = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR], 
                                           outputs=[self.scoreP, self.scoreN])
        if verbose:
            logger.debug('Architecture of MLP Ranker built finished, summarized below: ')
            logger.debug('Input dimension: %d' % edim)
            logger.debug('Hidden dimension: %d' % args.hidden)
            logger.debug('Total number of parameters used in the model: %d' % self.num_params)

    def update_params(self, grads, learn_rate):
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * grad, borrow=True)

    @staticmethod
    def save(fname, model):
        with file(fname, 'wb') as fout:
            cPickle.dump(model, fout)

    @staticmethod
    def load(fname):
        with file(fname, 'rb') as fin:
            model = cPickle.load(fin)
        return model

ranker = MLPRanker()
end_time = time.time()
logger.debug('Time used to build MLPRanker: %f seconds.' % (end_time-start_time))
# Define negative/positive sampling ratio
# Begin training
# Using AdaGrad learning algorithm
learn_rate = args.rate
batch_size = args.size
fudge_factor = 1e-6
logger.debug('MLPRanker.params: {}'.format(ranker.params))
hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in ranker.params]
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
# Build final training and test matrices
trainL = np.zeros((train_size, edim), dtype=floatX)
trainR = np.zeros((train_size, edim), dtype=floatX)
trainNR = np.zeros((train_size, edim), dtype=floatX)

testL = np.zeros((test_size, edim), dtype=floatX)
testR = np.zeros((test_size, edim), dtype=floatX)
testNR = np.zeros((test_size, edim), dtype=floatX)

for i in xrange(train_size):
    trainL[i, :] = np.mean(train_pairs_set[i][0], axis=0)
    trainR[i, :] = np.mean(train_pairs_set[i][1], axis=0)
    ni = train_neg_index[i]
    trainNR[i, :] = np.mean(train_pairs_set[ni][1], axis=0)

for i in xrange(test_size):
    testL[i, :] = np.mean(test_pairs_set[i][0], axis=0)
    testR[i, :] = np.mean(test_pairs_set[i][1], axis=0)
    testNR[i, :] = np.mean(test_pairs_set[i][2], axis=0)
end_time = time.time()
logger.debug('Time used to generate negative training and test pairs: %f seconds.' % (end_time-start_time))

nepoch = 50
try: 
    start_time = time.time()
    num_batch = train_size / batch_size
    for i in xrange(nepoch):
        logger.debug('-' * 50)
        # Looper over training instances
        total_cost, total_count = 0.0, 0
        # Compute the number of batches
        logger.debug('Batch size = %d' % batch_size)
        logger.debug('Total number of batches: %d' % num_batch)
        # Using GPU computation
        for j in xrange(num_batch):
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in ranker.params]
            hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in ranker.params]
            r = ranker.compute_cost_and_gradient(trainL[j*batch_size : (j+1)*batch_size], 
                                                 trainR[j*batch_size : (j+1)*batch_size],
                                                 trainL[j*batch_size : (j+1)*batch_size],
                                                 trainNR[j*batch_size : (j+1)*batch_size])
            inst_grads, costs, score_p, score_n = r[:-3], r[-3], r[-2], r[-1]
            for tot_grad, hist_grad, inst_grad in zip(total_grads, hist_grads, inst_grads):
                tot_grad += inst_grad
                hist_grad += np.square(inst_grad)
            total_cost += np.sum(costs) * batch_size
            total_count += np.sum(score_p >= score_n)
            # AdaGrad updating
            for tot_grad, hist_grad in zip(total_grads, hist_grads):
                tot_grad /= fudge_factor + np.sqrt(hist_grad)
            ranker.update_params(total_grads, learn_rate)
        # Final updating on the result of training instances
        if num_batch * batch_size < train_size:
            total_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in ranker.params]
            hist_grads = [np.zeros(param.get_value(borrow=True).shape, dtype=floatX) for param in ranker.params]
            r = ranker.compute_cost_and_gradient(trainL[num_batch*batch_size:],
                                                 trainR[num_batch*batch_size:],
                                                 trainL[num_batch*batch_size:],
                                                 trainNR[num_batch*batch_size:])
            inst_grads, costs, score_p, score_n = r[:-3], r[-3], r[-2], r[-1]
            for tot_grad, hist_grad, inst_grad in zip(total_grads, hist_grads, inst_grads):
                tot_grad += inst_grad
                hist_grad += np.square(inst_grad)
            total_cost += np.sum(costs) * (train_size-num_batch*batch_size)
            total_count += np.sum(score_p >= score_n)
            # AdaGrad updating
            for tot_grad, hist_grad in zip(total_grads, hist_grads):
                tot_grad /= fudge_factor + np.sqrt(hist_grad)
            ranker.update_params(total_grads, learn_rate)
        # Compute training error
        train_accuracy = total_count / float(train_size)
        # Reporting after each training epoch
        logger.debug('Training @ %d epoch, total cost = %f, accuracy = %f' % (i, total_cost, train_accuracy))
        if train_accuracy > highest_train_accuracy: highest_train_accuracy = train_accuracy
        # Testing after each training epoch
        total_cost, total_count = 0.0, 0        
        score_p, score_n = ranker.show_scores(testL, testR, testL, testNR)
        tmp_cost = 1-score_p+score_n
        tmp_cost[tmp_cost <= 0.0] = 0.0
        total_cost += np.sum(tmp_cost)
        total_count += np.sum(score_p >= score_n)
        test_accuracy = total_count / float(test_size)
        logger.debug('Test accuracy: %f' % test_accuracy)
        logger.debug('Test total cost: %f' % total_cost)
        if test_accuracy > highest_test_accuracy: highest_test_accuracy = test_accuracy
    end_time = time.time()
    logger.debug('Time used for training: %f minutes.' % ((end_time-start_time)/60))
    # Final total test
    start_time = time.time()
    test_cost, total_count = 0.0, 0
    score_p, score_n = ranker.show_scores(testL, testR, testL, testNR)
    tmp_cost = 1-score_p+score_n
    tmp_cost[tmp_cost <= 0.0] = 0.0
    total_cost += np.sum(tmp_cost)
    total_count += np.sum(score_p >= score_n)
    test_accuracy = total_count / float(test_size)
    end_time = time.time()
    logger.debug('Time used for testing: %f seconds.' % (end_time-start_time))
    logger.debug('Test accuracy: %f' % test_accuracy)
    logger.debug('Test total cost: %f' % total_cost)
except:
    logger.debug('!!!Error!!!')
    traceback.print_exc(file=sys.stdout)
    logger.debug('-' * 60)
finally:            
    logger.debug('Highest Training Accuracy: %f' % highest_train_accuracy)
    logger.debug('Highest Test Accuracy: %f' % highest_test_accuracy)
    logger.debug('Saving existing model and parameters...')
    logger.debug('Saving the model: rankerMatchRanker-{}.pkl.'.format(args.name))
    MLPRanker.save('rankerMatchRanker-{}.pkl'.format(args.name), ranker)
