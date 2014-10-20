#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2014-10-06 13:03:51
# @Author  : Han Zhao (han.zhao@uwaterloo.ca)
# @Link    : https://github.com/KeiraZhao
# @Version : $Id$

import os
import numpy as np
import theano
import theano.tensor as T
import time
import cPickle
import logging

import config
import utils
from utils import floatX
from activations import Activation
from mlp import HiddenLayer
from logistic import SoftmaxLayer, LogisticLayer
from score import ScoreLayer

logger = logging.getLogger(__name__)

class GrCNNEncoder(object):
    '''
    (Binary) Gated Recursive Convolutional Neural Network Encoder.
    '''
    def __init__(self, config=None, verbose=True):
        '''
        @config: GRCNNConfiger. Configer used to set the architecture of GRCNNEncoder.
        ''' 
        if verbose: logger.debug('Building Gated Recursive Convolutional Neural Network Encoder...')
        # Make theano symbolic tensor for input and model parameters
        self.input = T.matrix(name='input', dtype=floatX)
        # Configure activation function
        self.act = Activation(config.activation)
        fan_in, fan_out = config.num_input, config.num_hidden
        # Initialize model parameters
        # Set seed of the random generator
        np.random.seed(config.random_seed)
        # Projection matrix U
        self.U = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_in+fan_out)),
                                      high=np.sqrt(6.0/(fan_in+fan_out)),
                                      size=(fan_in, fan_out)), dtype=floatX),
                    name='U', borrow=True)
        self.hidden0 = T.dot(self.input, self.U)
        # W^l, W^r, parameters used to construct the central hidden representation
        self.Wl = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_out+fan_out)),
                                      high=np.sqrt(6.0/(fan_out+fan_out)),
                                      size=(fan_out, fan_out)), dtype=floatX),
                    name='W_l', borrow=True)
        self.Wr = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(fan_out+fan_out)),
                                      high=np.sqrt(6.0/(fan_out+fan_out)),
                                      size=(fan_out, fan_out)), dtype=floatX),
                    name='W_r', borrow=True)
        self.Wb = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='Wb', borrow=True)
        # G^l, G^r, parameters used to construct the three-way coefficients
        self.Gl = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(3+fan_out)),
                                      high=np.sqrt(6.0/(3+fan_out)),
                                      size=(fan_out, 3)), dtype=floatX),
                    name='G_l', borrow=True)
        self.Gr = theano.shared(value=np.asarray(
                    np.random.uniform(low=-np.sqrt(6.0/(3+fan_out)),
                                      high=np.sqrt(6.0/(3+fan_out)),
                                      size=(fan_out, 3)), dtype=floatX),
                    name='G_r', borrow=True)
        self.Gb = theano.shared(value=np.zeros(3, dtype=floatX), name='Gb', borrow=True)
        # Save all the parameters into one batch
        self.params = [self.U, self.Wl, self.Wr, self.Wb, self.Gl, self.Gr, self.Gb]
        # Length of the time sequence
        self.nsteps = self.input.shape[0]
        self.pyramids, _ = theano.scan(fn=self._step_prop, 
                                    sequences=T.arange(self.nsteps-1),
                                    outputs_info=[self.hidden0],
                                    n_steps=self.nsteps-1)
        self.output = self.pyramids[-1][0].dimshuffle('x', 0)
        # Compression -- Encoding function
        self.compress = theano.function(inputs=[self.input], outputs=self.output)
        if verbose:
            logger.debug('Finished constructing the structure of grCNN Encoder: ')
            logger.debug('Size of the input dimension: %d' % fan_in)
            logger.debug('Size of the hidden dimension: %d' % fan_out)
            logger.debug('Activation function: %s' % config.activation)

    def _step_prop(self, iter, current_level):
        '''
        @current_level: Input matrix at current level. The first dimension corresponds to 
        the timestamp while the second dimension corresponds to the dimension of hidden representation
        '''
        # Build shifted matrix, due to the constraints of Theano.scan, we have to keep the shape of the
        # input and output matrix
        left_current_level = current_level[:self.nsteps-iter-1]
        right_current_level = current_level[1:self.nsteps-iter]
        # Compute temporary central hidden representation, of size Txd, but we only care about the first
        # T-1 rows, i.e., we only focus on the (T-1)xd sub-matrix.
        central_current_level = self.act.activate(T.dot(left_current_level, self.Wl) + 
                                                  T.dot(right_current_level, self.Wr) + 
                                                  self.Wb)
        # Compute gating function, of size Tx3. Again, due to the internal limitation of Theano.scan, we cannot
        # reduce the size of the matrix and have to keep the same size, but actually we only want the first (T-1)x3
        # sub-matrix.
        current_gates = T.nnet.softmax(T.dot(left_current_level, self.Gl) + 
                                       T.dot(right_current_level, self.Gr) + 
                                       self.Gb)
        left_gate, central_gate, right_gate = current_gates[:, 0], current_gates[:, 1], current_gates[:, 2]
        # Reshape for broadcasting
        left_gate = left_gate.dimshuffle(0, 'x')
        central_gate = central_gate.dimshuffle(0, 'x')
        right_gate = right_gate.dimshuffle(0, 'x')
        # Build next level of hidden representation using soft combination,
        # matrix of size (T-1)xd
        next_level = left_gate * left_current_level + \
                     right_gate * right_current_level + \
                     central_gate * central_current_level
        return T.set_subtensor(current_level[:self.nsteps-iter-1], next_level)

    def _step_prop_reduce(self, current_level):
        '''
        @current_level: Input matrix at current level. The first dimension corresponds to 
        the timestamp while the second dimension corresponds to the dimension of hidden representation

        Reduced version of level propagation, much more memory and time efficient implementation, but cannot
        be used inside theano.scan because theano.scan requires that the input and output through timestamps should
        have the same shape.
        '''
        # Build shifted matrix, due to the constraints of Theano.scan, we have to keep the shape of the
        # input and output matrix
        right_current_level = current_level[1:]
        left_current_level = current_level[:-1]
        # Compute temporary central hidden representation, of size Txd, but we only care about the first
        # T-1 rows, i.e., we only focus on the (T-1)xd sub-matrix.
        central_current_level = self.act.activate(T.dot(left_current_level, self.Wl) + 
                                                  T.dot(right_current_level, self.Wr) + 
                                                  self.Wb)
        # Compute gating function, of size Tx3. Again, due to the internal limitation of Theano.scan, we cannot
        # reduce the size of the matrix and have to keep the same size, but actually we only want the first (T-1)x3
        # sub-matrix.
        current_gates = T.nnet.softmax(T.dot(left_current_level, self.Gl) + 
                                       T.dot(right_current_level, self.Gr) + 
                                       self.Gb)
        left_gate, central_gate, right_gate = current_gates[:, 0], current_gates[:, 1], current_gates[:, 2]
        # Reshape for broadcasting
        left_gate = left_gate.dimshuffle(0, 'x')
        central_gate = central_gate.dimshuffle(0, 'x')
        right_gate = right_gate.dimshuffle(0, 'x')
        # Build next level of hidden representation using soft combination,
        # matrix of size (T-1)xd
        next_level = left_gate * left_current_level + \
                     right_gate * right_current_level + \
                     central_gate * central_current_level
        return next_level


class GrCNN(object):
    '''
    (Binary) Gated Recursive Convolutional Neural Network Classifier, with GRCNN as the 
    encoder part and MLP as the classifier part.
    '''
    def __init__(self, config=None, verbose=True):
        self.encoder = GrCNNEncoder(config, verbose)
        # Link two parts
        self.params = self.encoder.params
        self.input = self.encoder.input
        self.hidden = self.encoder.output
        # Activation function
        self.act = Activation(config.activation)
        # MLP Component
        self.hidden_layer = HiddenLayer(self.hidden, (config.num_hidden, config.num_mlp), act=Activation(config.hiddenact))
        self.compressed_hidden = self.hidden_layer.output
        # Dropout regularization
        srng = T.shared_randomstreams.RandomStreams(config.random_seed)
        mask = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hidden.shape)
        self.compressed_hidden *= T.cast(mask, floatX)
        # Accumulate model parameters
        self.params += self.hidden_layer.params
        # Softmax Component
        self.softmax_layer = SoftmaxLayer(self.compressed_hidden, (config.num_mlp, config.num_class))
        self.raw_output = self.softmax_layer.output
        self.pred = self.softmax_layer.pred
        self.params += self.softmax_layer.params
        # Compute the total number of parameters in this model
        self.num_params_encoder = config.num_input * config.num_hidden + \
                                  config.num_hidden * config.num_hidden * 2 + \
                                  config.num_hidden + \
                                  config.num_hidden * 3 * 2 + \
                                  3
        self.num_params_classifier = config.num_hidden * config.num_mlp + \
                                     config.num_mlp + \
                                     config.num_mlp * config.num_class + \
                                     config.num_class
        self.num_params = self.num_params_encoder + self.num_params_classifier
        # Build target function 
        self.truth = T.ivector(name='label')
        self.learn_rate = T.scalar(name='learning rate')
        self.cost = self.softmax_layer.NLL_loss(self.truth)
        # Build computational graph and compute the gradient of the target 
        # function with respect to model parameters
        self.gradparams = T.grad(self.cost, self.params)
        # Updates formula for stochastic gradient descent algorithm
        self.updates = []
        for param, gradparam in zip(self.params, self.gradparams):
            self.updates.append((param, param-self.learn_rate*gradparam))
        # Compile theano function
        self.objective = theano.function(inputs=[self.input, self.truth], outputs=self.cost)
        self.predict = theano.function(inputs=[self.input], outputs=self.pred)
        # Compute the gradient of the objective function with respect to the model parameters
        self.compute_cost_and_gradient = theano.function(inputs=[self.input, self.truth], outputs=self.gradparams+[self.cost])
        # Output function for debugging purpose
        self.show_hidden = theano.function(inputs=[self.input, self.truth], outputs=self.hidden)
        self.show_compressed_hidden = theano.function(inputs=[self.input, self.truth], outputs=self.compressed_hidden)
        self.show_output = theano.function(inputs=[self.input, self.truth], outputs=self.raw_output)
        if verbose:
            logger.debug('Architecture of GrCNN built finished, summarized as below: ')
            logger.debug('Input dimension: %d' % config.num_input)
            logger.debug('Hidden dimension inside GrCNNEncoder pyramid: %d' % config.num_hidden)
            logger.debug('Hidden dimension of MLP: %d' % config.num_mlp)
            logger.debug('Number of target classes: %d' % config.num_class)
            logger.debug('Number of parameters in encoder part: %d' % self.num_params_encoder)
            logger.debug('Number of parameters in classifier part: %d' % self.num_params_classifier)
            logger.debug('Number of total parameters in this model: %d' % self.num_params)

    def train(self, instance, label):
        '''
        @instance: np.ndarray. Two dimension matrix which corresponds to a time sequence.
        The first dimension along the matrix represents the time dimension while the second 
        dimension along the matrix represents the embedding dimension.
        @label: np.ndarray. 1 dimensional array of int as labels.
        @learn_rate: np.scalar. Learning rate of the stochastic gradient descent algorithm.
        '''
        cost = self.objective(instance, label)
        return cost

    def update_params(self, grads, learn_rate):
        '''
        @grads: [np.ndarray]. List of numpy.ndarray for updating the model parameters.
        @learn_rate: scalar. Learning rate.
        '''
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * grad, borrow=True)

    @staticmethod
    def save(fname, model):
        '''
        @fname: String. Filename to store the model.
        @model: GrCNN. An instance of GrCNN classifier to be saved.
        '''
        with file(fname, 'wb') as fout:
            cPickle.dump(model, fout)

    @staticmethod
    def load(fname):
        '''
        @fname: String. Filename to load the model.
        '''
        with file(fname, 'rb') as fin:
            model = cPickle.load(fin)
        return model


class GrCNNMatcher(object):
    '''
    (Binary) Gated Recursive Convolutional Neural Network for Matching task, 
    with two GrCNNEncoders as the encoder part and logistic regression as the 
    classifier part.
    '''
    def __init__(self, config=None, verbose=True):
        # Construct two GrCNNEncoders for matching two sentences
        self.encoderL = GrCNNEncoder(config, verbose)
        self.encoderR = GrCNNEncoder(config, verbose)
        # Link two parts
        self.params = []
        self.params += self.encoderL.params
        self.params += self.encoderR.params
        self.inputL = self.encoderL.input
        self.inputR = self.encoderR.input
        # Get output of two GrCNNEncoders
        self.hiddenL = self.encoderL.output
        self.hiddenR = self.encoderR.output
        # Activation function
        self.act = Activation(config.activation)
        # MLP Component
        self.hidden = T.concatenate([self.hiddenL, self.hiddenR], axis=1)
        self.hidden_layer = HiddenLayer(self.hidden, (2*config.num_hidden, config.num_mlp), act=Activation(config.hiddenact))
        self.compressed_hidden = self.hidden_layer.output
        # Accumulate parameters
        self.params += self.hidden_layer.params
        # Dropout parameter
        srng = T.shared_randomstreams.RandomStreams(config.random_seed)
        mask = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hidden.shape)
        self.compressed_hidden *= T.cast(mask, floatX)
        # Use concatenated vector as input to the logistic regression classifier
        self.logistic_layer = LogisticLayer(self.compressed_hidden, config.num_mlp)
        self.output = self.logistic_layer.output
        self.pred = self.logistic_layer.pred
        # Accumulate parameters
        self.params += self.logistic_layer.params
        # Compute the total number of parameters in this model
        self.num_params_encoder = config.num_input * config.num_hidden + \
                                  config.num_hidden * config.num_hidden * 2 + \
                                  config.num_hidden + \
                                  config.num_hidden * 3 * 2 + \
                                  3
        self.num_params_encoder *= 2
        self.num_params_classifier = 2 * config.num_hidden * config.num_mlp + \
                                     config.num_mlp + \
                                     config.num_mlp + 1
        self.num_params = self.num_params_encoder + self.num_params_classifier
        # Build target function
        self.truth = T.ivector(name='label')
        self.learn_rate = T.scalar(name='learning rate')
        self.cost = self.logistic_layer.NLL_loss(self.truth)
        # Build computational graph and compute the gradient of the target function
        # with respect to model parameters
        self.gradparams = T.grad(self.cost, self.params)
        # Updates formula for stochastic descent algorithm
        self.updates = []
        for param, gradparam in zip(self.params, self.gradparams):
            self.updates.append((param, param-self.learn_rate*gradparam))
        # Compile theano function
        self.objective = theano.function(inputs=[self.inputL, self.inputR, self.truth], outputs=self.cost)
        self.predict = theano.function(inputs=[self.inputL, self.inputR], outputs=self.pred)
        # Compute the gradient of the objective function with respect to the model parameters
        self.compute_cost_and_gradient = theano.function(inputs=[self.inputL, self.inputR, self.truth], 
                                                outputs=self.gradparams+[self.cost, self.pred])
        # Output function for debugging purpose
        self.show_hidden = theano.function(inputs=[self.inputL, self.inputR, self.truth], outputs=self.hidden)
        self.show_compressed_hidden = theano.function(inputs=[self.inputL, self.inputR, self.truth], outputs=self.compressed_hidden)
        self.show_output = theano.function(inputs=[self.inputL, self.inputR, self.truth], outputs=self.output)
        if verbose:
            logger.debug('Architecture of GrCNNMatcher built finished, summarized below: ')
            logger.debug('Input dimension: %d' % config.num_input)
            logger.debug('Hidden dimension inside GrCNNMatcher pyramid: %d' % config.num_hidden)
            logger.debug('Hidden dimension of MLP: %d' % config.num_mlp)            
            logger.debug('Number of parameters in encoder part: %d' % self.num_params_encoder)
            logger.debug('Number of parameters in classifier part: %d' % self.num_params_classifier)
            logger.debug('Number of total parameters in this model: %d' % self.num_params)            

    def update_params(self, grads, learn_rate):
        '''
        @grads: [np.array]. List of numpy.ndarray for updating the model parameters.
        @learn_rate: scalar. Learning rate.
        '''
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * grad, borrow=True)

    @staticmethod
    def save(fname, model):
        '''
        @fname: String. Filename to store the model.
        @model: GrCNNMatcher. An instance of GrCNNMatcher to be saved.
        '''
        with file(fname, 'wb') as fout:
            cPickle.dump(model, fout)

    @staticmethod
    def load(fname):
        '''
        @fname: String. Filename to load the model.
        '''
        with file(fname, 'rb') as fin:
            model = cPickle.load(fin)
        return model

# Derive from GrCNNMatcher and only change the output of the last layer
class GrCNNMatchScorer(GrCNNMatcher):
    '''
    Gated Recursive Convolutional Neural Network for matching task. The last 
    layer of the model includes a linear layer for regression.
    '''
    def __init__(self, config=None, verbose=True):
        # Call initialization in parent method to build architecture
        super(GrCNNMatchScorer, self).__init__(config, verbose)
        # Override, use concatenated vector as input to the score layer
        self.score_layer = ScoreLayer(self.compressed_hidden, config.num_mlp) 
        self.output = self.score_layer.output
        # Revise the parameters of this model
        self.params = self.params[:-2]
        self.params += self.score_layer.params
    
class GrCNNMatchRanker(object):
    '''
    Use two GrCNNMatchScorer to score the positive and negative pairs.
    '''
    def __init__(self, config=None, verbose=True):
        # Build two components for scoring pairs of sentences
        self.p_scorer = GrCNNMatchScorer(config, verbose)
        self.n_scorer = GrCNNMatchScorer(config, verbose)
        # Extract scores
        self.p_score = self.p_scorer.output
        self.n_score = self.n_scorer.output
        # Stack parameters
        self.params = []
        self.params += self.p_scorer.params
        self.params += self.n_scorer.params
        # Compute the total number of parameters in the model
        self.num_params = self.p_scorer.num_params + self.n_scorer.num_params
        # Build prediction function 
        self.pred = self.p_scorer.output >= self.n_scorer.output
        # Build target function 
        self.cost = T.mean(T.maximum(0, 1.0 - self.p_score + self.n_score))
        # Construct gradients of the target function with respect to the model parameters
        self.gradparams = T.grad(self.cost, self.params)
        # Build actual functions
        self.objective = theano.function(inputs=[self.p_scorer.inputL, self.p_scorer.inputR, 
                                                 self.n_scorer.inputL, self.n_scorer.inputR],
                                         outputs=self.cost)
        self.predict = theano.function(inputs=[self.p_scorer.inputL, self.p_scorer.inputR, 
                                               self.n_scorer.inputL, self.n_scorer.inputR],
                                       outputs=self.pred)
        # Compute the gradient of the objective function with respect to the model parameters
        self.compute_cost_and_gradient = theano.function(inputs=[self.p_scorer.inputL, self.p_scorer.inputR, 
                                                                 self.n_scorer.inputL, self.n_scorer.inputR],
                                                         outputs=self.gradparams+[self.cost, self.pred])
        if verbose:
            logger.debug('Architecture of GrCNNMatchRanker built finished, summarized below: ')
            logger.debug('Input dimension: %d' % config.num_input)
            logger.debug('Hidden dimension inside GrCNNMatcherRanker pyramid: %d' % config.num_hidden)
            logger.debug('Hidden dimension MLP: %d' % config.num_mlp)
            logger.debug('There are 4 GrCNN Encoders used in model.')
            logger.debug('There are 2 MLP Hidden layers used in model.')
            logger.debug('There is 2 Linear score layers used in model.')
            logger.debug('There is 1 output unit used in model.')
            logger.debug('Total number of parameters used in model: %d' % self.num_params)

    def update_params(self, grads, learn_rate):
        '''
        @grads: [np.ndarray]. List of numpy.ndarray for updating the model parameters.
        @learn_rate: scalar. Learning rate.
        '''
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * grad, borrow=True)

    @staticmethod
    def save(fname, model):
        '''
        @fname: String. Filename to store the model.
        @model: GrCNNMatchRanker. An instance of GrCNNMatchRanker to be saved.
        '''
        with file(fname, 'wb') as fout:
            cPickle.dump(model, fout)

    @staticmethod
    def load(fname):
        '''
        @fname: String. Filename to load the model.
        '''
        with file(fname, 'rb') as fin:
            model = cPickle.load(fin)
        return model

