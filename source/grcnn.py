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
        # Scale factor for initializing parameters
        self.scale = config.scale
        # Make theano symbolic tensor for input and model parameters
        self.input = T.matrix(name='GrCNN Encoder input', dtype=floatX)
        # Configure activation function
        self.act = Activation(config.activation)
        fan_in, fan_out = config.num_input, config.num_hidden
        # Initialize model parameters
        # Set seed of the random generator
        np.random.seed(config.random_seed)
        # Projection matrix U
        # Initialize all the matrices using orthogonal matrices
        U_val = np.random.uniform(low=-1.0, high=1.0, size=(fan_in, fan_out))
        U_val = U_val.astype(floatX)
        U_val *= self.scale
        self.U = theano.shared(value=U_val, name='U', borrow=True)
        self.hidden0 = T.dot(self.input, self.U)

        # W^l, W^r, parameters used to construct the central hidden representation
        Wl_val = np.random.uniform(low=-1.0, high=1.0, size=(fan_out, fan_out))
        Wl_val = Wl_val.astype(floatX)
        Wl_val, _, _ = np.linalg.svd(Wl_val)
        # Wl_val *= self.scale
        self.Wl = theano.shared(value=Wl_val, name='W_l', borrow=True)

        Wr_val = np.random.uniform(low=-1.0, high=1.0, size=(fan_out, fan_out))
        Wr_val = Wr_val.astype(floatX)
        Wr_val, _, _ = np.linalg.svd(Wr_val)
        # Wr_val *= self.scale
        self.Wr = theano.shared(value=Wr_val, name='W_r', borrow=True)
        
        self.Wb = theano.shared(value=np.zeros(fan_out, dtype=floatX), name='Wb', borrow=True)
        
        # G^l, G^r, parameters used to construct the three-way coefficients
        Gl_val = np.random.uniform(low=-1.0, high=1.0, size=(fan_out, 3))
        Gl_val = Gl_val.astype(floatX)
        self.Gl = theano.shared(value=Gl_val, name='G_l', borrow=True)

        Gr_val = np.random.uniform(low=-1.0, high=1.0, size=(fan_out, 3))
        Gr_val = Gr_val.astype(floatX)
        self.Gr = theano.shared(value=Gr_val, name='G_r', borrow=True)

        self.Gb = theano.shared(value=np.zeros(3, dtype=floatX), name='Gb', borrow=True)
        # Save all the parameters into one batch
        self.params = [self.U, self.Wl, self.Wr, self.Wb, self.Gl, self.Gr, self.Gb]
        # Length of the time sequence
        self.nsteps = self.input.shape[0]
        self.pyramids, _ = theano.scan(fn=self._step_prop, 
                                    sequences=T.arange(self.nsteps-1),
                                    non_sequences=self.nsteps,
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

    def _step_prop(self, iter, current_level, nsteps):
        '''
        @current_level: Input matrix at current level. The first dimension corresponds to 
        the timestamp while the second dimension corresponds to the dimension of hidden representation
        '''
        # Build shifted matrix, due to the constraints of Theano.scan, we have to keep the shape of the
        # input and output matrix
        left_current_level = current_level[:nsteps-iter-1]
        right_current_level = current_level[1:nsteps-iter]
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
        return T.set_subtensor(current_level[:nsteps-iter-1], next_level)

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

    def encode(self, inputM):
        '''
        @input: Theano symbol matrix. Compress the input matrix into output vector.
        '''
        hidden = T.dot(inputM, self.U)
        # Length of the time sequence
        nsteps = inputM.shape[0]
        pyramids, _ = theano.scan(fn=self._step_prop, 
                                    sequences=T.arange(nsteps-1),
                                    non_sequences=nsteps,
                                    outputs_info=[hidden],
                                    n_steps=nsteps-1)
        output = pyramids[-1][0].dimshuffle('x', 0)
        return output


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
class GrCNNMatchScorer(object):
    '''
    Gated Recursive Convolutional Neural Network for matching task. The last 
    layer of the model includes a linear layer for regression.
    '''
    def __init__(self, config=None, verbose=True):
        # Construct two GrCNNEncoders for matching two sentences
        self.encoderL = GrCNNEncoder(config, verbose)
        self.encoderR = GrCNNEncoder(config, verbose)
        # Link the parameters of two parts
        self.params = []
        # self.params += self.encoderL.params
        # self.params += self.encoderR.params

        # Build three kinds of inputs:
        # 1, inputL, inputR. This pair is used for computing the score after training
        # 2, inputPL, inputPR. This part is used for training positive pairs
        # 3, inputNL, inputNR. This part is used for training negative pairs
        self.inputL = self.encoderL.input
        self.inputR = self.encoderR.input
        # Positive
        self.inputPL = T.matrix(name='inputPL', dtype=floatX)
        self.inputPR = T.matrix(name='inputPR', dtype=floatX)
        # Negative
        self.inputNL = T.matrix(name='inputNL', dtype=floatX)
        self.inputNR = T.matrix(name='inputNR', dtype=floatX)
        # Linking input-output mapping
        self.hiddenL = self.encoderL.output
        self.hiddenR = self.encoderR.output
        # Positive 
        self.hiddenPL = self.encoderL.encode(self.inputPL)
        self.hiddenPR = self.encoderR.encode(self.inputPR)
        # Negative
        self.hiddenNL = self.encoderL.encode(self.inputNL)
        self.hiddenNR = self.encoderR.encode(self.inputNR)
        # Activation function
        self.act = Activation(config.activation)
        # MLP Component
        self.hidden = T.concatenate([self.hiddenL, self.hiddenR], axis=1)
        self.hiddenP = T.concatenate([self.hiddenPL, self.hiddenPR], axis=1)
        self.hiddenN = T.concatenate([self.hiddenNL, self.hiddenNR], axis=1)
        # Build hidden layer
        self.hidden_layer = HiddenLayer(self.hidden, (2*config.num_hidden, config.num_mlp), act=Activation(config.hiddenact))
        self.compressed_hidden = self.hidden_layer.output
        self.compressed_hiddenP = self.hidden_layer.encode(self.hiddenP)
        self.compressed_hiddenN = self.hidden_layer.encode(self.hiddenN)
        # Accumulate parameters
        self.params += self.hidden_layer.params
        # Dropout parameter
        srng = T.shared_randomstreams.RandomStreams(config.random_seed)
        mask = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hidden.shape)
        maskP = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hiddenP.shape)
        maskN = srng.binomial(n=1, p=1-config.dropout, size=self.compressed_hiddenN.shape)
        self.compressed_hidden *= T.cast(mask, floatX)
        self.compressed_hiddenP *= T.cast(maskP, floatX)
        self.compressed_hiddenN *= T.cast(maskN, floatX)
        # Score layers
        self.score_layer = ScoreLayer(self.compressed_hidden, config.num_mlp)
        # Reset the weights of the score layer
        self.score_layer.W.set_value(np.asarray(np.random.uniform(low=-1.0, high=1.0, size=config.num_mlp),
                            dtype=floatX))
        self.output = self.score_layer.output
        self.scoreP = self.score_layer.encode(self.compressed_hiddenP)
        self.scoreN = self.score_layer.encode(self.compressed_hiddenN)
        # Accumulate parameters
        self.params += self.score_layer.params
        # Build cost function
        self.cost = T.mean(T.maximum(0, 1.0 - self.scoreP + self.scoreN))
        # Construct the gradient of the cost function with respect to the model parameters
        self.gradparams = T.grad(self.cost, self.params)
        # Compute the total number of parameters in the model
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
        # Build class methods
        self.score = theano.function(inputs=[self.inputL, self.inputR], outputs=self.output)
        self.compute_cost_and_gradient = theano.function(inputs=[self.inputPL, self.inputPR, 
                                                                 self.inputNL, self.inputNR],
                                                         outputs=self.gradparams+[self.cost, self.scoreP, self.scoreN])
        self.show_scores = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR], 
                                           outputs=[self.scoreP, self.scoreN])
        self.show_hiddens = theano.function(inputs=[self.inputPL, self.inputPR, self.inputNL, self.inputNR],
                                            outputs=[self.hiddenP, self.hiddenN])
        if verbose:
            logger.debug('Architecture of GrCNNMatchScorer built finished, summarized below: ')
            logger.debug('Input dimension: %d' % config.num_input)
            logger.debug('Hidden dimension inside GrCNNMatchScorer pyramid: %d' % config.num_hidden)
            logger.debug('Hidden dimension MLP: %d' % config.num_mlp)
            logger.debug('There are 2 GrCNNEncoders used in model.')
            logger.debug('Total number of parameters used in the model: %d' % self.num_params)

    def update_params(self, grads, learn_rate):
        '''
        @grads: [np.ndarray]. List of numpy.ndarray for updating the model parameters.
        @learn_rate: scalar. Learning rate.
        '''
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * p, borrow=True)

    @staticmethod
    def save(fname, model):
        '''
        @fname: String. Filename to store the model.
        @model: GrCNNMatchScorer. An instance of GrCNNMatchScorer to be saved.
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

