#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-08-06 13:49:13
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : 0.0

import os, sys
import cPickle
import unittest
import time
sys.path.append('../source/')
from pprint import pprint
import theano
import theano.tensor as T
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from utils import floatX
from cnn import ConvNet
from logistic import SoftmaxLayer
from mlp import MLP
from wordvec import WordEmbedding
from config import CNNConfiger, MLPConfiger

class TestSentiment(unittest.TestCase):
	def setUp(self):
		'''
		Load training and test texts and labels 
		in sentiment analysis task, preprocessing.
		'''
		np.random.seed(1991)
		senti_train_set_filename = '../data/sentiment_train_txt.txt'
		senti_train_label_filename = '../data/sentiment_train_label.txt'
		senti_test_set_filename = '../data/sentiment_test_txt.txt'
		senti_test_label_filename = '../data/sentiment_test_label.txt'
		embedding_filename = '../data/wiki_embeddings.txt'
		# Load training/test data sets and wiki-embeddings
		word_embedding = WordEmbedding(embedding_filename)
		with file(senti_train_set_filename) as fin:
			senti_train_txt = fin.readlines()
		with file(senti_test_set_filename) as fin:
			senti_test_txt = fin.readlines()
		# Store original text representation
		self.senti_train_txt = senti_train_txt
		self.senti_test_txt = senti_test_txt
		# Word-vector representation
		self.senti_train_label = np.loadtxt(senti_train_label_filename, dtype=np.int32)
		self.senti_test_label = np.loadtxt(senti_test_label_filename, dtype=np.int32)
		train_size = len(senti_train_txt)
		test_size = len(senti_test_txt)
		# Check size
		assert train_size == self.senti_train_label.shape[0]
		assert test_size == self.senti_test_label.shape[0]
		pprint('Training size: %d' % train_size)
		pprint('Test size: %d' % test_size)
		# Compute word embedding
		self.senti_train_set = np.zeros((train_size, word_embedding.embedding_dim()), dtype=floatX)
		self.senti_test_set = np.zeros((test_size, word_embedding.embedding_dim()), dtype=floatX)
		# Embedding for training set
		for i, sent in enumerate(senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			# pprint('Trainging set, Number of words in sentence %d: %d' % (i, len(words)))
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_train_set[i, :] = np.mean(vectors, axis=0)
		# Embedding for test set
		for i, sent in enumerate(senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words]
			# pprint('Test set, Number of words in sentence %d: %d' % (i, len(words)))
			vectors = np.asarray([word_embedding.wordvec(word) for word in words])
			self.senti_test_set[i, :] = np.mean(vectors, axis=0)
		# Shuffle training and test data set
		train_rand_index = np.random.permutation(train_size)
		test_rand_index = np.random.permutation(test_size)
		self.senti_train_set = self.senti_train_set[train_rand_index, :]
		self.senti_test_set = self.senti_test_set[test_rand_index, :]
		self.senti_train_label = self.senti_train_label[train_rand_index]
		self.senti_test_label = self.senti_test_label[test_rand_index]
		# Store data
		self.train_size = train_size
		self.test_size = test_size

	@unittest.skip('Without sparsity constraint: accuracy = 0.70717 \
					With sparsity constraint: accuracy = 0.706623')
	def testSoftmax(self):
		'''
		Sentiment analysis task for sentence representation using 
		softmax classifier.
		'''
		input = T.matrix(name='input')
		label = T.ivector(name='label')
		learning_rate = T.scalar(name='learning rate')
		num_in, num_out = 50, 2
		softmax = SoftmaxLayer(input, (num_in, num_out))
		lambdas = 1e-5
		cost = softmax.NLL_loss(label) + lambdas * softmax.L2_loss()
		# cost = softmax.NLL_loss(label)
		params = softmax.params
		gradparams = T.grad(cost, params)
		updates = []
		for param, gradparam in zip(params, gradparams):
			updates.append((param, param-learning_rate*gradparam))
		objective = theano.function(inputs=[input, label, learning_rate], outputs=cost, updates=updates)
		# Training
		nepoch = 5000
		start_time = time.time() 
		for i in xrange(nepoch):
			rate = 2.0 / ((1.0 + i/500) ** 2)
			func_value = objective(self.senti_train_set, self.senti_train_label, rate)
			prediction = softmax.predict(self.senti_train_set)
			accuracy = np.sum(prediction == self.senti_train_label) / float(self.train_size)
			pprint('epoch %d, cost = %f, accuracy = %f' % (i, func_value, accuracy))
		end_time = time.time()
		pprint('Time used to train the softmax classifier: %f minutes' % ((end_time-start_time)/60))
		# Test
		prediction = softmax.predict(self.senti_test_set)
		accuracy = np.sum(prediction == self.senti_test_label) / float(self.test_size)
		pprint('Test accuracy: %f' % accuracy)

	@unittest.skip('accuracy @ sigmoid = 0.7099 \
					accuracy @ tanh = 0.7104 \
					accuracy @ ReLU = 0.715380')
	def testMLP(self):
		'''
		Sentiment analysis task for sentence representation using MLP, 
		with one hidden layer and one softmax layer.
		'''
		conf_filename = './sentiment_mlp.conf'
		start_time = time.time()
		configer = MLPConfiger(conf_filename)
		mlpnet = MLP(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build the architecture of MLP: %f seconds.' % (end_time-start_time))
		# Training
		start_time = time.time()
		for i in xrange(configer.nepoch):
			cost, accuracy = mlpnet.train(self.senti_train_set, self.senti_train_label)
			pprint('epoch %d, cost = %f, accuracy = %f' % (i, cost, accuracy))
		end_time = time.time()
		pprint('Time used for training MLP network on Sentiment analysis task: %f minutes.' % ((end_time-start_time)/60))
		# Test
		prediction = mlpnet.predict(self.senti_test_set)
		accuracy = np.sum(prediction == self.senti_test_label) / float(self.test_size)
		pprint('Test accuracy: %f' % accuracy)

	@unittest.skip('accuracy @ sigmoid = 0.7104 \
					accuracy @ tanh = 0.7159 \
					accuracy @ ReLU = 0.6584~0.7094')
	def testCNN(self):
		conf_filename = './sentiment_cnn.conf'
		# Build the architecture of CNN
		start_time = time.time()
		configer = CNNConfiger(conf_filename)
		convnet = ConvNet(configer, verbose=True)
		end_time = time.time()
		pprint('Time used to build the architecture of CNN: %f seconds' % (end_time-start_time))
		# Training
		learn_rate = 0.5
		batch_size = configer.batch_size
		num_batches = self.train_size / batch_size
		start_time = time.time()
		for i in xrange(configer.nepoch):
			right_count = 0
			# rate = learn_rate
			rate = learn_rate / (i/100+1)
			for j in xrange(num_batches):
				minibatch = self.senti_train_set[j*batch_size : (j+1)*batch_size, :]
				minibatch = minibatch.reshape((batch_size, 1, configer.image_row, configer.image_col))
				label = self.senti_train_label[j*batch_size : (j+1)*batch_size]
				cost, accuracy = convnet.train(minibatch, label, rate)
				prediction = convnet.predict(minibatch)
				right_count += np.sum(label == prediction)
				# pprint('Epoch %d, batch %d, cost = %f, local accuracy: %f' % (i, j, cost, accuracy))
			accuracy = right_count / float(self.train_size)
			pprint('Epoch %d, overall accuracy: %f' % (i, accuracy))
			ConvNet.save('./sentiment.cnn', convnet)
		end_time = time.time()
		pprint('Time used to train CNN on Sentiment analysis task: %f minutes.' % ((end_time-start_time)/60))
		# Test
		num_batches = self.test_size / batch_size
		right_count = 0
		for i in xrange(num_batches):
			minibatch = self.senti_test_set[i*batch_size : (i+1)*batch_size, :]
			minibatch = minibatch.reshape((batch_size, 1, configer.image_row, configer.image_col))
			label = self.senti_test_label[i*batch_size : (i+1)*batch_size]
			prediction = convnet.predict(minibatch)
			right_count += np.sum(prediction == label)
		test_accuracy = right_count / float(self.test_size)
		pprint('Test set accuracy: %f' % test_accuracy)

	def testBoGNB(self):
		'''
		Test on sentiment analysis task using Naive Bayes classifier 
		with Bag-of-Word feature vectors.
		'''
		wordlist = []
		# Preprocessing of original txt data set
		for i, sent in enumerate(self.senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words if len(word) > 2]
			wordlist.extend(words)
		for i, sent in enumerate(self.senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words if len(word) > 2]
			wordlist.extend(words)
		word_dict = set(wordlist)
		word2index = dict(zip(word_dict, range(len(word_dict))))
		# Build BoG feature
		train_size = len(self.senti_train_txt)
		test_size = len(self.senti_test_txt)
		pprint('Training set size: %d' % train_size)
		pprint('Test set size: %d' % test_size)
		train_feat = np.zeros((train_size, len(word_dict)), dtype=np.float)
		test_feat = np.zeros((test_size, len(word_dict)), dtype=np.float)
		# Using binary feature
		start_time = time.time()
		for i, sent in enumerate(self.senti_train_txt):
			words = sent.split()
			words = [word.lower() for word in words if len(word) > 2]
			indices = map(lambda x: word2index[x], words)
			train_feat[i, indices] = 1.0
		for i, sent in enumerate(self.senti_test_txt):
			words = sent.split()
			words = [word.lower() for word in words if len(word) > 2]
			indices = map(lambda x: word2index[x], words)
			test_feat[i, indices] = 1.0
		end_time = time.time()
		pprint('Finished building training and test feature matrix, time used: %f seconds.' % (end_time-start_time))
		pprint('Classification using Bernoulli Naive Bayes classifier: ')
		clf = BernoulliNB()
		# clf = LogisticRegression()
		clf.fit(train_feat, self.senti_train_label)
		train_pred_label = clf.predict(train_feat)
		train_acc = np.sum(train_pred_label == self.senti_train_label) / float(train_size)
		pprint('Training accuracy = %f' % train_acc)
		pred_label = clf.predict(test_feat)
		acc = np.sum(pred_label == self.senti_test_label) / float(test_size)
		pprint('Accuracy: %f' % acc)
		train_pos_count = np.sum(self.senti_train_label == 1)
		train_neg_count = np.sum(self.senti_train_label == 0)
		test_pos_count = np.sum(self.senti_test_label == 1)
		test_neg_count = np.sum(self.senti_test_label == 0)
		pprint('Positive count in training set: %d' % train_pos_count)
		pprint('Negative count in training set: %d' % train_neg_count)
		pprint('Ratio: pos/neg = %f' % (float(train_pos_count) / train_neg_count))
		pprint('Positive count in test set: %d' % test_pos_count)
		pprint('Negative count in test set: %d' % test_neg_count)
		pprint('Ratio: pos/neg = %f' % (float(test_pos_count) / test_neg_count))


if __name__ == '__main__':
	unittest.main()


