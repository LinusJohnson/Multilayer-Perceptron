import numpy as np
import theano
from src.BayesianOptimizer import BayesianOptimization
from src.recover import recover

#bounds = {'numLayers': (1, 40), 'layerSize': (75, 500), 'firstLayerDropout': (0.5, 1), 
#         'dropout': (0.1, 1), 'lambda1': (0, 1), 'lambda2': (0, 1), 'c': (0, 5), 
#         'SPDhits': (0.499999, 0.500001), 'isolationc': (0.499999, 0.500001), 'IPSig': (0.499999, 0.500001),
#         'IP': (0.499999, 0.500001), 'rho': (0, 0.99), 'epsilon': (1e-10, 7e-3)}

#bounds = {'numLayers': (8, 40), 'layerSize': (200, 500), 'firstLayerDropout': (0.7, 1), 
#      'dropout': (0.1, 1), 'lambda1': (0, 0.5), 'lambda2': (0, 0.5), 'c': (0, 6), 
#      'SPDhits': (0.499999, 0.500001), 'rho': (0.3, 0.99), 'epsilon': (1e-9, 1e-1)}

bounds = {'numLayers': (15, 30), 'layerSize': (250, 550), 'firstLayerDropout': (0.8, 0.95), 
      'dropout': (0.5, 1), 'lambda1': (0, 0.3), 'lambda2': (0.3, 1), 'c': (0.4, 6), 
      'SPDhits': (0.499999, 0.500001), 'rho': (0.3, 0.99), 'epsilon': (1e-13, 1e-9)}
recoverData = recover()

# to make sure that c = 0 gets explored since it turns off max-norm but lies next to really heavy max-norm regularising values
#explore = {'numLayers': [15], 'layerSize': [200], 'firstLayerDropout': [0.8], 'dropout': [0.5], 'lambda1': [0], 'lambda2': [0], 'c': [0], 'SPDhits': [0.499999], 'rho': [0.95], 'epsilon': [1e-9]}

rng = np.random.RandomState(123)
kFold = 4
trainSetSize = 67553 * (1 - (1./kFold))
print "trainSetSize", trainSetSize
batchSize = int(trainSetSize / 10.)
print "batchSize", batchSize 
patience = 12000
numEpochs = int((patience / 10.) * 15)
print "numEpochs", numEpochs
BayesianOpt = BayesianOptimization(bounds, rng = rng, optAlg = "adaDelta", batchSize = batchSize, kFold = kFold, numEpochs = numEpochs, 
                                   validationFrequency = 1, patience = patience, visualize = True, test = False)
BayesianOpt.initialize(*recoverData)
#BayesianOpt.explore(explore)
BayesianOpt.minimize(numInitPoints = 0, numIter = 15)

print "Run completed"