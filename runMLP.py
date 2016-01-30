from src.helpFunctions import loadTrainingData, generateSubmission, initalizeModel, trainWithEarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import theano
#import sys
#sys.setrecursionlimit(10000)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#debug = False
#theano.config.allow_gc = True
#theano.config.floatX = 'float32'
#if debug:
#	theano.config.exception_verbosity = 'high'
#	theano.config.optimizer = 'None'
#	theano.config.mode = 'DebugMode'
#else:
	#import os
	#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
	#theano.config.exception_verbosity = 'low'
	#theano.config.optimizer = 'fast_compile'
	#theano.config.optimizer = 'fast_run'
	#theano.config.mode = 'FAST_RUN'
	#theano.config.optimizer_including = local_remove_all_assert
	#theano.config.profile = False
	#theano.config.profile_memory = False
#theano.config.optimizer_including="local_ultra_fast_sigmoid"
#hyperparameters = {}
#hyperparameters["numLayers"] = 11 #25 #14
#hyperparameters["firstLayerDropout"] = 0.51110627032139799 #0.7
#hyperparameters["dropout"] = 0.18899487803764103 #0.3
#hyperparameters["layerSize"] = 244 #400 #200
#hyperparameters["lambda1"] = 0.29533832244856095 #0
#hyperparameters["lambda2"] = 0.91251732052714851 #0.001
#hyperparameters["c"] = 4.3596601831217345 #1.5
#hyperparameters["learningRate"] = 1.0246657903697596 #0.0001
#hyperparameters["rho"] = 0.53696556759404956 #0.95 # 0.95
#hyperparameters["epsilon"] = 0.0087182807500100819 #1e-6 # 1e-9
hyperparameters = {'c': 5, 'SPDhits': 0.49, 'dropout': 0.9, 'firstLayerDropout': 0.85, 
'layerSize': 1, 'rho': 0.75, 'epsilon': 1e-10, 'numLayers': 1, 'lambda1': 1e-4, 'lambda2': 0.5}

hyperparameters["isolationc"] = 1
hyperparameters["IPSig"] = 1
hyperparameters["IP"] = 1

rng = np.random.RandomState(123)
hyperparameters["rng"] = rng
modelName = ""
if 'numLayers' in hyperparameters: modelName += "%iL" % int(hyperparameters['numLayers'])
if 'layerSize' in hyperparameters: modelName += "%iLS" % int(hyperparameters['layerSize'])
if 'firstLayerDropout' in hyperparameters: modelName += "%.2fFD" % hyperparameters['firstLayerDropout']
if 'dropout' in hyperparameters: modelName += "%gD" % hyperparameters['dropout']
if 'SPDhits' in hyperparameters: modelName += "%iSPD" % np.round(hyperparameters['SPDhits'])
if 'isolationc' in hyperparameters: modelName += "%iISO" % np.round(hyperparameters['isolationc'])
if 'IPSig' in hyperparameters: modelName += "%iIPS" % np.round(hyperparameters['IPSig'])
if 'IP' in hyperparameters: modelName += "%iIP" % np.round(hyperparameters['IP'])
if 'c' in hyperparameters: modelName += "%.2fC" % hyperparameters['c']
if 'lambda1' in hyperparameters: modelName += "%.2fL1" % hyperparameters['lambda1']
if 'lambda2' in hyperparameters: modelName += "%.2fL2" % hyperparameters['lambda2']
if 'rho' in hyperparameters: modelName += "%.2fRHO" % hyperparameters['rho']
if 'epsilon' in hyperparameters: modelName += "%gEPS" % hyperparameters['epsilon']
print "Model name: " + modelName
variables = ['FlightDistance','FlightDistanceError', 'LifeTime', 'VertexChi2','pt','dira','DOCAone',
             'DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2', 'isolationa', 'isolationb', 'isolationd',
             'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 
             'p2_IsoBDT', 'p0_track_Chi2Dof','p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_pt','p0_p','p0_eta',
             'p0_IP','p0_IPSig','p1_pt','p1_p', 'p1_eta','p1_IP','p1_IPSig','p2_pt','p2_p','p2_eta','p2_IP',
             'p2_IPSig']
#if SPDhits: variables += ['SPDhits']
if np.round(hyperparameters.pop('SPDhits')): variables += ['SPDhits']
if np.round(hyperparameters.pop('isolationc')): variables += ['isolationc']
if np.round(hyperparameters.pop('IPSig')): variables += ['IPSig']
if np.round(hyperparameters.pop('IP')): variables += ['IP']
#if np.round(hyperparameters.pop('isolationc')): variables += ['isolationc']
#if np.round(hyperparameters.pop('IPSig')): variables += ['IPSig']
#if np.round(hyperparameters.pop('IP')): variables += ['IP']
hyperparameters["numFeatures"] = len(variables)

data = loadTrainingData(variables, rng, valPart = 0.3, verbose = True)
optAlgs = ["RMSProp", "adaDelta", "vSGDfd"]
patience = 200 #12000
validationFrequency = 1
batchSize = int(data[0].get_value(borrow=True).shape[0]) #int(data[0].get_value(borrow=True).shape[0] / 6.) #5000 #300
numTrainBatches = data[0].get_value(borrow=True).shape[0] / batchSize
numEpochs = 200 #int(patience / numTrainBatches) * 15

hyperparameters["optAlg"] = optAlgs[1]
mlp = initalizeModel(**hyperparameters)
trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, name = modelName)