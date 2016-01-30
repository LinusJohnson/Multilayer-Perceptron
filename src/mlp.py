import theano
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode
import numpy as np
from layer import Layer
import theano.d3viz as d3v
#from optimizationAlgorithms import maxNormReg, iRpropPlus, RMSProp, vSGDfd
#import optimizationAlgorithms
class MLP(object):
    def __init__(self, rng, activations, layerSizes, probabilities, lambda1, lambda2, c, optAlg, learningRate, rho, epsilon, momentum):
        self.rng = rng
        self.layers = [Layer(self, self.rng, activations[i], layerSizes[i], layerSizes[i + 1], probabilities[i]) for i in range(len(activations))]
        self.thetas = [layer.theta for layer in self.layers]
        self.alphas = [layer.alpha for layer in self.layers if layer.activation == T.nnet.relu]
        self.gammas = [layer.gamma for layer in self.layers]
        self.betas = [layer.beta for layer in self.layers]
        self.means = [layer.mean for layer in self.layers]
        self.stds = [layer.std for layer in self.layers]
        self.params = [layer.params for layer in self.layers]
        self.extraParams = [extraParam for layer in self.layers for extraParam in layer.extraParams]
        self.dropoutStream = theano.sandbox.rng_mrg.MRG_RandomStreams(seed = self.rng.randint(1, 2147462579), use_cuda = True)
        self.lambda1 =  theano.shared(value = np.float32(lambda1), name='lambda1')
        self.lambda2 = theano.shared(value = np.float32(lambda2), name='lambda2')
        self.c = theano.shared(value = np.float32(c), name='c')
        self.optAlg = optAlg
        self.learningRate = learningRate
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        
    def updateExtraParams(self):
        self.extraParams = [extraParam for layer in self.layers for extraParam in layer.extraParams]
    
    def forwardProp(self, x, training = True):
        for layer in self.layers:
            x = layer.output(self.dropout(x, layer, training), training)
        return x
    
    def dropout(self, x, layer, training):
        p = layer.dropoutProbability
        if training and p != 0:
            x *= T.cast(self.dropoutStream.binomial(size = x.shape, n = 1, p = p), theano.config.floatX)
            x /= (p + 1e-9)
        return x
    
    def binaryCrossEntropyCostFunction(self, hyp, y):
        reg1 = self.lambda1 * sum(map(lambda theta: T.sum(T.abs_(theta), acc_dtype = theano.config.floatX, dtype = theano.config.floatX), self.thetas))
        reg2 = self.lambda2 * sum(map(lambda theta: T.sum(T.sqr(theta), acc_dtype = theano.config.floatX, dtype = theano.config.floatX), self.thetas))
        return T.mean(T.nnet.binary_crossentropy(hyp, y)) + reg1 + reg2
        #return T.mean(-(T.log(hyp) * y + (1.0 - y) * T.log(1.0 - hyp)), acc_dtype = theano.config.floatX, dtype = theano.config.floatX) + reg1 + reg2
        # could make false negatives more expensive by multiplying the cost function part for positive y:s by for example 10
    
    def train(self, Xtrain, Ytrain, batchSize):
        X = T.matrix("X")
        Y = T.matrix("Y")
        index = T.iscalar("index")
        cost = self.binaryCrossEntropyCostFunction(self.forwardProp(X, True), Y)
        inputs = [index]

        if self.optAlg.lower() == "irpropplus":
            updates = iRpropPlus(cost, self.params, self.c)
        elif self.optAlg.lower() == "vsgdfd":
            slowStart = T.fscalar("slowStart")
            updates = vSGDfd(cost, self.params, batchSize, self.layers[0].theta.shape.eval()[1], self.c, slowStart)
            inputs.append(slowStart)
        elif self.optAlg.lower() == "rmsprop":
            updates = RMSProp(cost, self.params, self.c, self.learningRate, self.rho, self.epsilon, self.momentum)
        elif self.optAlg.lower() == "adadelta":
            updates = adaDelta(cost, self.params, self.c, self.rho, self.epsilon)
        else:
            updates = RMSProp(cost, self.params, self.c, self.learningRate, self.rho, self.epsilon, self.momentum)

        train = theano.function(inputs = inputs,
                                outputs = theano.Out(cost, borrow = True),
                                updates = updates + self.extraParams,
                                name = "train",
                                givens = {X: Xtrain[index * batchSize:(index + 1) * batchSize,:],
                                          Y: Ytrain[index * batchSize:(index + 1) * batchSize,:]},
                                allow_input_downcast = True
                                #,mode = theano.compile.MonitorMode(pre_func = inspect_inputs, post_func = inspect_outputs)
                                #,mode = NanGuardMode(nan_is_error = True, inf_is_error = True, big_is_error = True)
                                #,mode = theano.compile.MonitorMode(post_func = detect_nan)
                                #,mode = theano.compile.MonitorMode(post_func = detect_inf)
                                ,profile = True
                                )
        #print theano.printing.debugprint(train)
        #theano.printing.pydotprint(train, outfile="symbolic_graph_of_nn_opt.svg", format='svg', var_with_name_simple=True)
        #d3v.d3viz(train, 'd3viz/mlp.html')
        return train

    def predict(self):
        dataToPredictOn = T.matrix("dataToPredictOn")
        predict = theano.function(inputs = [dataToPredictOn], 
                               outputs = self.forwardProp(dataToPredictOn, False),
                               allow_input_downcast = True,
                               name = "predict")
        #theano.printing.pydotprint(predict, outfile="symbolic_graph_of_nn_forward_opt.png", var_with_name_simple=True)
        return predict

    # Should the validation error be done in minibatches?

    #def validationError(self, Xval, Yval, batchSize):
    #    X = T.matrix("X")
    #    Y = T.matrix("Y")
    #    index = T.iscalar("index")
    #    return theano.function(inputs = [index], 
    #                           outputs = theano.Out(self.binaryCrossEntropyCostFunction(self.forwardProp(X, False), Y), borrow = True),
    #                           givens = {X: Xval[index * batchSize:(index + 1) * batchSize,:], 
    #                                     Y: Yval[index * batchSize:(index + 1) * batchSize,:]},
    #                           allow_input_downcast = True,
    #                           name = "validationError")
    
    def validationError(self, Xval, Yval):
        X = T.matrix("X")
        Y = T.matrix("Y")
        return theano.function(inputs = [], 
                               outputs = self.binaryCrossEntropyCostFunction(self.forwardProp(X, False), Y),
                               allow_input_downcast = True,
                               givens = {X: Xval, 
                                         Y: Yval},
                               name = "validationError")
    
    def saveModel(self, name, folder, verbose = True):
        #ParamsToBeSaved = {}
        #for param in self.params:
        #    ParamsToBeSaved.update({"layer_" + str(i) + "_": np.asarray(param.T.eval()) for i in xrange(len(param))})
        ParamsToBeSaved = {"layer_" + str(i) + "_thetas": np.asarray(self.thetas[i].eval()) for i in xrange(len(self.thetas))}
        ParamsToBeSaved.update({"layer_" + str(i) + "_alphas": np.asarray(self.alphas[i].T.eval()) for i in xrange(len(self.alphas))})
        ParamsToBeSaved.update({"layer_" + str(i) + "_betas": np.asarray(self.betas[i].eval()) for i in xrange(len(self.betas))})
        ParamsToBeSaved.update({"layer_" + str(i) + "_gammas": np.asarray(self.gammas[i].eval()) for i in xrange(len(self.gammas))})
        ParamsToBeSaved.update({"layer_" + str(i) + "_means": np.asarray(self.means[i].eval()) for i in xrange(len(self.means))})
        ParamsToBeSaved.update({"layer_" + str(i) + "_stds": np.asarray(self.stds[i].eval()) for i in xrange(len(self.stds))})
        np.savez(folder + 'savedParams' + name, **ParamsToBeSaved)
        if verbose:
            print "Controlling that the saved parameters are correct"
            savedParams = np.load(folder + 'savedParams' + name + '.npz')
            contr = True
            for i in xrange(len(self.alphas)):
                if not (savedParams["layer_" + str(i) + "_alphas"].T == np.asarray(self.alphas[i].eval())).all():
                    contr = False
            for i in xrange(len(self.betas)):
                if not (savedParams["layer_" + str(i) + "_betas"] == np.asarray(self.betas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_gammas"] == np.asarray(self.gammas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_thetas"] == np.asarray(self.thetas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_means"] == np.asarray(self.means[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_stds"] == np.asarray(self.stds[i].eval())).all(): 
                    contr = False
            print "The saving was successful" if contr else "Something went wrong"

    def loadModel(self, name, folder, verbose = True):
        savedParams = np.load(folder + 'savedParams' + name + '.npz')
        for i in xrange(len(self.layers)):
            layer = self.layers[i]
            if layer.activation == T.nnet.relu:
                layer.alpha.set_value(savedParams["layer_" + str(i) + "_alphas"].T)
            layer.beta.set_value(savedParams["layer_" + str(i) + "_betas"])
            layer.gamma.set_value(savedParams["layer_" + str(i) + "_gammas"])
            layer.theta.set_value(savedParams["layer_" + str(i) + "_thetas"])
            layer.mean.set_value(savedParams["layer_" + str(i) + "_means"])
            layer.std.set_value(savedParams["layer_" + str(i) + "_stds"])
        if verbose:
            contr = True
            for i in xrange(len(self.alphas)):
                if not (savedParams["layer_" + str(i) + "_alphas"].T == np.asarray(self.alphas[i].eval())).all():
                    contr = False
            for i in xrange(len(self.betas)):
                if not (savedParams["layer_" + str(i) + "_betas"] == np.asarray(self.betas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_gammas"] == np.asarray(self.gammas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_thetas"] == np.asarray(self.thetas[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_means"] == np.asarray(self.means[i].eval())).all() \
                or not (savedParams["layer_" + str(i) + "_stds"] == np.asarray(self.stds[i].eval())).all(): 
                    contr = False
            print "The loading was successful" if contr else "Something went wrong"

def maxNormReg(param, c, epsilon):
    # max-norm regularization as described in the dropout paper
    norms = T.sqrt(T.sum(T.sqr(param), axis = 0, keepdims = True, acc_dtype = theano.config.floatX))
    param *= T.clip(norms, 0., c) / (norms + epsilon) + T.eq(c, 0.)
    return param

def iRpropPlus(cost, params, c, positiveStep = np.float32(1.2), negativeStep = np.float32(0.5), maxStep = np.float32(50), minStep = np.float32(1e-6)):
    updates = []
    for layerParams in params:
        for param in layerParams:
            lastParamGradSign = theano.shared(param.get_value(borrow=True) * 0.)
            lastParamDelta = theano.shared(param.get_value(borrow=True) * 0.1)
            lastCost = theano.shared(np.float32(np.inf))

            gradient = T.grad(cost = cost, wrt = param, disconnected_inputs = 'raise')
            change = T.sgn(lastParamGradSign * gradient)
            changePos = T.gt(change, 0.0).astype(theano.config.floatX)
            changeNeg = T.lt(change, 0.0).astype(theano.config.floatX)
            changeZero = T.eq(change, 0.0).astype(theano.config.floatX)
            costInc = T.gt(cost, lastCost).astype(theano.config.floatX)
            newParam = param - changePos * T.sgn(gradient) * T.minimum(lastParamDelta * positiveStep, maxStep) \
                       + changeNeg * costInc * lastParamGradSign * lastParamDelta \
                       - changeZero * T.sgn(gradient) * lastParamDelta

            # max-norm regularization
            newParam = maxNormReg(newParam, c, epsilon)

            newLastParamDelta = changePos * T.minimum(lastParamDelta * positiveStep, maxStep).astype(theano.config.floatX) \
                                + changeNeg * T.maximum(lastParamDelta * negativeStep, minStep).astype(theano.config.floatX) \
                                + changeZero * lastParamDelta.astype(theano.config.floatX)
            newLastParamGradSign = changePos * T.sgn(gradient).astype(theano.config.floatX) \
                                   + changeNeg * 0 \
                                   + changeZero * T.sgn(gradient).astype(theano.config.floatX)
            updates.append((param, newParam))
            updates.append((lastParamDelta, newLastParamDelta))
            updates.append((lastParamGradSign, newLastParamGradSign))
    updates.append((lastCost, cost))
    return updates

def RMSProp(cost, params, c, lr, rho = 0.9, epsilon = 1e-9, momentum = 0.9):
    updates = []
    for layerParams in params:
        for param in layerParams:
            acc = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), borrow = True, name = "acc")
            vel = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), borrow = True, name = "vel")
            gradient = T.grad(cost = cost, wrt = param, disconnected_inputs = 'raise')

            newAcc = rho * acc + (T.cast(1., theano.config.floatX) - rho) * gradient ** 2
            scaledGradient = lr * gradient / T.sqrt(newAcc + epsilon)
            newVel = momentum * vel - scaledGradient
            newParam = param + momentum * newVel - scaledGradient

            # max-norm regularization
            newParam = maxNormReg(newParam, c, epsilon)

            updates.append((acc, newAcc))
            updates.append((vel, newVel))
            updates.append((param, newParam))
    return updates

def adaDelta(cost, params, c, rho = 0.9, epsilon = 1e-9):
    updates = []
    for layerParams in params:
        for param in layerParams:
            sqrGradAcc = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), borrow = True, name = "sqrGradAcc")
            sqrUpdAcc = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), borrow = True, name = "sqrUpdAcc")
            gradient = T.grad(cost = cost, wrt = param, disconnected_inputs = 'raise')

            newSqrGradAcc = rho * sqrGradAcc + (T.cast(1., theano.config.floatX) - rho) * T.sqr(gradient)
            update = - (T.sqrt(sqrUpdAcc + epsilon) / T.sqrt(newSqrGradAcc + epsilon)) * gradient
            newSqrUpdAcc = rho * sqrUpdAcc + (T.cast(1., theano.config.floatX) - rho) * T.sqr(update)
            newParam = param + update

            # max-norm regularization
            newParam = maxNormReg(newParam, c, epsilon)

            updates.append((param, newParam))
            updates.append((sqrGradAcc, newSqrGradAcc))
            updates.append((sqrUpdAcc, newSqrUpdAcc))
    return updates

# not working
def vSGDfd(cost, params, batchSize, numFeatures, c, slowStart, epsilon = 1e-9):
    updates = []
    for layerParams in params:
        for param in layerParams:
            gbar = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), name = "gbar")
            vbar = theano.shared((np.ones_like(param.get_value(borrow=True)) * (numFeatures / 10.)).astype(theano.config.floatX), name = "vbar")
            hbar = theano.shared(np.zeros_like(param.get_value(borrow=True)).astype(theano.config.floatX), name = "hbar")
            vhbar = theano.shared((np.ones_like(param.get_value(borrow=True)) * epsilon).astype(theano.config.floatX), name = "vhbar")
            tau = theano.shared(((np.ones_like(param.get_value(borrow=True)) + epsilon) * 2.).astype(theano.config.floatX), name = "tau")
            
            gradient = T.grad(cost = cost, wrt = param, disconnected_inputs = 'raise') # cost is already mini-batch mean so the gradient is so as well
            #shiftedParams = theano.clone(param, replace = {param: param + gbar})
            #shiftedGradient = T.grad(cost = None, wrt = shiftedParams, known_grads = {shiftedParams: gradient}, disconnected_inputs = 'raise')
            
            #printGradient = theano.printing.Print("gradient:")
            #gradient = printGradient(gradient)

            #tempParam = param.get_value()
            #param.set_value(param.get_value() + gbar.get_value())
            #shiftedParam = param + gbar
            #e = theano.gof.Env([param], [gradient])
            #e.replace(param, shiftedParam)
            #shiftedGradientFunc = getGrad(cost, X, Y, Xvalues, Yvalues, index, param)
            #shiftedGradient = shiftedGradientFunc(shiftedParam)
            #shiftedGradient = T.grad()
            #printShiftedGradient = theano.printing.Print("shiftedGradient:")
            #shiftedGradient = printShiftedGradient(np.asarray(shiftedGradient))

            # using the RMSProp hessian approximation
            hessian = T.sqrt(vbar)
            newtau = T.clip(T.gt(T.abs_(gradient - gbar), 2. * T.sqrt(vbar - gbar ** 2)) + 
                            T.gt(T.abs_(hessian - hbar), 2. * T.sqrt(vhbar - hbar ** 2)), 0., 1.) + tau

            fract = 1. / (newtau + epsilon)

            # running mean gradient
            newgbar = (1. - fract) * gbar + fract * gradient
            #newgbar += (T.le(newgbar, epsilon) * T.ge(newgbar, 0.) - T.lt(newgbar, 0.) * T.ge(newgbar, -epsilon)) * epsilon
            #newgbar = T.switch(T.lt(newgbar, epsilon), epsilon, newgbar)

            # running mean squared gradient
            newvbar = (1. - fract) * vbar + fract * T.sqr(gradient) #+ gStabConstant
            # running mean Hessian
            newhbar = (1. - fract) * hbar + fract * hessian
            # for numerical stability
            #newhbar += (T.le(newhbar, epsilon) * T.ge(newhbar, 0.) - T.lt(newhbar, 0.) * T.ge(newhbar, -epsilon)) * epsilon
            #newhbar = T.switch(T.lt(newhbar, epsilon), epsilon, newhbar)
            # running mean squared Hessian
            newvhbar = (1. - fract) * vhbar + fract * T.sqr(hessian)

            gbarSqr = T.sqr(newgbar)
            hDivvh = (newhbar / (newvhbar + epsilon))
            rest = ((batchSize * gbarSqr) / (newvbar + (batchSize - 1) * gbarSqr + epsilon))
            lr = hDivvh * rest

            newParam = param - T.abs_(lr) * gradient
            newTau = (1 - gbarSqr / (newvbar + epsilon)) * newtau + 1

            # max-norm regularization
            newParam = maxNormReg(newParam, c, epsilon)

            # if slow starting the params won't be updated
            updates.append((param, T.cast(newParam * (1 - slowStart) + slowStart * param, theano.config.floatX)))
            updates.append((gbar, newgbar))
            updates.append((vbar, newvbar))
            updates.append((hbar, newhbar))
            updates.append((vhbar, newvhbar))
            updates.append((tau, newTau))
    return updates


def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]
    
def detect_inf(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and np.isinf(output[0]).any()):
            print '*** inf detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], np.random.RandomState) and np.isnan(output[0]).any()):
            print '*** nan detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [np.asarray(input[0]) for input in fn.inputs]
            print 'Outputs: %s' % [np.asarray(output[0]) for output in fn.outputs]
            break