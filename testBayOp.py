import numpy as np
import theano
from src.BayesianOptimizer import BayesianOptimization

# bounds = {'numLayers': (1, 40), 'layerSize': (1, 500), 'firstLayerDropout': (0.4, 1), 
# 		'dropout': (0.00001, 1), 'lambda1': (0, 1), 'lambda2': (0, 1), 'c': (0, 6), 
# 		'SPDhits': (0.499999, 0.500001), 'isolationc': (0.499999, 0.500001), 'IPSig': (0.499999, 0.500001),
# 		'IP': (0.499999, 0.500001), 'rho': (0, 0.99), 'epsilon': (1e-10, 1e-2)}

#initializeTest = {-0.97: {'numLayers': 14, 'layerSize': 128, 'firstLayerDropout': 0.8,
#				'dropout': 0.7, 'lambda1': 0, 'lambda2': 0, 'c': 3.5, 'SPDhits': 0.499999, 'isolationc': 0.500001, 
#				'IPSig': 0.500001, 'IP': 0.500001, 'learningRate': 0.001, 'rho': 0.95, 'epsilon': 1e-7}}
#from src.recover import recover
#testRecoverDict = recover()

# adaDelta test

bounds = {'numLayers': (8, 40), 'layerSize': (200, 500), 'firstLayerDropout': (0.6, 1), 
		'dropout': (0.1, 1), 'lambda1': (0, 0.5), 'lambda2': (0, 0.5), 'c': (0, 6), 
		'SPDhits': (0.499999, 0.500001), 'rho': (0.35, 0.99), 'epsilon': (1e-9, 1e-1)}

lowerAndUpperBounds = {key: [bounds[key][0], bounds[key][1]] for key, value in bounds.iteritems()}

rng = np.random.RandomState(123)
kFold = 4
trainSetSize = 67553 * (1 - (1./kFold))
batchSize = int(trainSetSize / 9.)
patience = 2
numEpochs = int((patience / 9.) * 15)
BayesianOpt = BayesianOptimization(bounds, rng = rng, optAlg = "adaDelta", batchSize = batchSize, kFold = kFold, numEpochs = numEpochs, validationFrequency = 1, patience = patience, visualize = True, test = True)
#BayesianOpt.initialize(testRecoverDict)
#BayesianOpt.explore(lowerAndUpperBounds)
BayesianOpt.minimize(numInitPoints = 2, numIter = 1)

# RMSprop test

bounds = {'numLayers': (8, 40), 'layerSize': (200, 500), 'firstLayerDropout': (0.6, 1), 
		'dropout': (0.1, 1), 'lambda1': (0, 0.5), 'lambda2': (0, 0.5), 'c': (0, 6), 
		'SPDhits': (0.499999, 0.500001), 'rho': (0.35, 0.99), 'momentum': (0.35, 0.99), 'learningRate': (0.0001, 1), 'epsilon': (1e-9, 1e-1)}

lowerAndUpperBounds = {key: [bounds[key][0], bounds[key][1]] for key, value in bounds.iteritems()}

rng = np.random.RandomState(123)
BayesianOpt = BayesianOptimization(bounds, rng = rng, optAlg = "RMSprop", batchSize = batchSize, kFold = kFold, numEpochs = numEpochs, validationFrequency = 1, patience = patience, visualize = True, test = True)
#BayesianOpt.initialize(testRecoverDict)
BayesianOpt.explore(lowerAndUpperBounds)
BayesianOpt.minimize(numInitPoints = 1, numIter = 1)

print "Test was successful"