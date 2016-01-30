import theano
from theano import tensor as T
import numpy as np
class Layer(object):
    def __init__(self, mlp, rng, activation, inputSize, outputSize, p):
        alphaInit = 0.25
        self.theta = theano.shared(value = self.initializeWeights(inputSize, outputSize, activation, alphaInit, rng), name='theta', borrow = True)
        self.beta = theano.shared(value = np.zeros((outputSize)).astype(theano.config.floatX), name='beta', borrow = True)
        self.gamma = theano.shared(value = np.ones((outputSize)).astype(theano.config.floatX), name='gamma', borrow = True)
        self.mean = theano.shared(value = np.zeros((outputSize)).astype(theano.config.floatX), name = 'mean', borrow = True)
        self.std = theano.shared(value = np.ones((outputSize)).astype(theano.config.floatX), name = 'std', borrow = True)
        if activation == T.nnet.relu:
            self.alpha = theano.shared(value = np.ones((1, outputSize)).astype(theano.config.floatX) * alphaInit, name = "alpha", borrow = True)
            self.params = [self.alpha, self.beta, self.gamma, self.theta]
        else:
            self.params = [self.beta, self.gamma, self.theta]
        self.extraParams = []
        self.activation = activation
        self.dropoutProbability = theano.shared(value = np.float32(p), name='dropoutProbability', borrow = True)
        self.mlp = mlp
        
    def initializeWeights(self, inputSize, outputSize, activation, alphaInit, rng):
        # MSRA initialization as in http://arxiv.org/pdf/1502.01852.pdf
        if activation == T.nnet.relu:
            std = np.sqrt(2 / (inputSize * (1 + alphaInit**2)))
            return rng.normal(loc = 0, scale = std, size=(inputSize, outputSize)).astype(theano.config.floatX)
        else:
            epsilon = np.sqrt(6 / (np.float32(inputSize) + np.float32(outputSize) + 1e-9) ).astype(theano.config.floatX) # for sigmoid
            return (rng.randn(inputSize, outputSize) * 2.0 * epsilon - epsilon).astype(theano.config.floatX)
    
    def output(self, input, training):
        # z = T.dot(input, self.theta[1:,:]) + self.theta[0,:] # removed bias due to using batch normalization where beta is used instead
        z = T.dot(input, self.theta)
        normedZ = self.BatchNormalization(z, training)
        return self.activation(normedZ, T.addbroadcast(self.alpha, 0)) if self.activation == T.nnet.relu else self.activation(normedZ)

    def BatchNormalization(self, input, training, rho = 0.9):
        if training:
            mean = T.mean(input, axis = 0, acc_dtype = theano.config.floatX, dtype = theano.config.floatX)
            #std = T.std(input, axis = 0)
            #size = T.cast(input.shape[0], theano.config.floatX)
            #mean = T.sum(input, axis = 0, acc_dtype = theano.config.floatX, dtype = theano.config.floatX) / size
            #std = T.sqrt((T.sum(T.sqr(input - mean), axis = 0, acc_dtype = theano.config.floatX, dtype = theano.config.floatX) / size))
            std = T.sqrt(T.mean(T.sqr(input - mean), axis = 0, acc_dtype = theano.config.floatX, dtype = theano.config.floatX) + 1e-9)
            self.extraParams = [(self.mean, rho * mean + (T.cast(1., theano.config.floatX) - rho) * self.mean), 
                                (self.std, rho * std + (T.cast(1., theano.config.floatX) - rho) * self.std)]
            self.mlp.updateExtraParams()
        else:
            mean = self.mean
            std = self.std
        return (input - mean) * (self.gamma / std) + self.beta