from src.helpFunctions import loadTrainingData, generateSubmission, initalizeModel, trainWithEarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import theano
#import sys
#sys.setrecursionlimit(10000)

debug = False
theano.config.allow_gc = True
theano.config.floatX = 'float32'
if debug:
	theano.config.exception_verbosity = 'high'
	theano.config.optimizer = 'None'
	theano.config.mode = 'DebugMode'
else:
	#import os
	#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
	theano.config.exception_verbosity = 'low'
	#theano.config.exception_verbosity = 'low'
	#theano.config.optimizer = 'fast_compile'
	theano.config.optimizer = 'fast_run'
	#theano.config.mode = 'Mode'
	#theano.config.optimizer_including = local_remove_all_assert
	theano.config.profile = False
	theano.config.profile_memory = False

#hyperparameters = {'c': 3.6698840769269192, 'SPDhits': 0.50000006834882049, 'isolationc': 0.50000052774716053, 'IP': 0.4999994679492728, 
#'dropout': 0.61199535616675316, 'firstLayerDropout': 0.66494650085852569, 'layerSize': 448.74304884515681, 'rho': 0.9, 
#'epsilon': 1e-10, 'numLayers': 10.93349338264288, 'lambda1': 0.0, 'lambda2': 0.23939490147676901, 'IPSig': 0.50000000813427736}
#hyperparameters = {'c': 4.930659585461231, 'SPDhits': 0.50000038509301892, 'isolationc': 0.50000039528356066, 'IP': 0.49999937659739307, 'dropout': 0.79348200223595711, 'firstLayerDropout': 0.62385278014346057, 'layerSize': 334.06248988500499, 'rho': 0.1703765229764346, 'epsilon': 1e-10, 'numLayers': 27.031396843789608, 'lambda1': 0.53877377671633664, 'lambda2': 0.21087760961525792, 'IPSig': 0.499999706548728}
hyperparameters = {'c': 5, 'SPDhits': 0.49, 'dropout': 0.5, 'firstLayerDropout': 0.85, 
'layerSize': 1500, 'rho': 0.75, 'epsilon': 1e-10, 'numLayers': 10, 'lambda1': 1e-4, 'lambda2': 0.5}

#hyperparameters = {'c': 10, 'SPDhits': 0.50000006834882049, 'isolationc': 0.50000052774716053, 'IP': 0.5, 
#'dropout': 0.61199535616675316, 'firstLayerDropout': 0.66494650085852569, 'layerSize': 448.74304884515681, 'rho': 0.48788114701003021, 
#'epsilon': 1e-10, 'numLayers': 10.93349338264288, 'lambda1': 0, 'lambda2': 0, 'IPSig': 0.50000000813427736}
#hyperparameters["learningRate"] = 0.001
#hyperparameters["momentum"] = 0.9

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
#if isolationc: variables += ['isolationc']
#if IPSig: variables += ['IPSig']
#if IP: variables += ['IP']
if np.round(hyperparameters.pop('SPDhits')): variables += ['SPDhits']
if np.round(hyperparameters.pop('isolationc')): variables += ['isolationc']
if np.round(hyperparameters.pop('IPSig')): variables += ['IPSig']
if np.round(hyperparameters.pop('IP')): variables += ['IP']
hyperparameters["numFeatures"] = len(variables)

data = loadTrainingData(variables, rng, valPart = 0.3, verbose = True)
optAlgs = ["RMSProp", "adaDelta", "vSGDfd"]
patience = 12000
validationFrequency = 1
batchSize = int(data[0].get_value(borrow=True).shape[0] / 12.)
#totNumSamples = 67553
numTrainBatches = data[0].get_value(borrow=True).shape[0] / batchSize
numEpochs = int(patience / numTrainBatches) * 15


hyperparameters["optAlg"] = optAlgs[1]
errorDict = {}
AUCDict = {}
mlp = initalizeModel(**hyperparameters)
score = trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, errorDict, AUCDict, name = modelName, visualize = True)

print "Score:", score
print "Max valAUC:", max(AUCDict["valAUC"])

plt.ioff()

fig1 = plt.figure(1)
fig1.canvas.set_window_title('Error/Epochs')
plt.title('Error/Epochs ' + hyperparameters["optAlg"])
plt.ylabel('Error')
plt.xlabel('Epochs')
plt.axis([0, numEpochs * numTrainBatches, min(errorDict["trainError"]), max(errorDict["trainError"])])
plt.plot(errorDict["epoch"], errorDict["trainError"], 'r-', errorDict["epoch"], errorDict["valError"], 'b-', label = 'errors')

fig2 = plt.figure(2)
fig2.canvas.set_window_title('AUC/Epochs')
plt.title('AUC/Epochs ' + hyperparameters["optAlg"])
plt.ylabel('AUC')
plt.xlabel('Epochs')
plt.axis([0, numEpochs * numTrainBatches, min(AUCDict["trainAUC"]), max(AUCDict["trainAUC"])])
plt.plot(AUCDict["epoch"], AUCDict["trainAUC"], 'r-', AUCDict["epoch"], AUCDict["valAUC"], 'b-', label = 'AUCs')

plt.show()

#hyperparameters["epsilon"] = 1e-9
#hyperparameters["rho"] = 0.9
# patience = 500
# numEpochs = int(patience / numTrainBatches) + 2
# hyperparameters["optAlg"] = optAlgs[0]
# errorDict = {}
# AUCDict = {}
# mlp = initalizeModel(**hyperparameters)
# score = trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, errorDict, AUCDict, name = modelName, visualize = True)
# #numTrainBatches = data[0].get_value(borrow=True).shape[0] / batchSize

# print "Score:", score
# print "Max valAUC:", max(AUCDict["valAUC"])

# fig = plt.figure(3)
# fig.canvas.set_window_title('Error/Epochs')
# plt.title('Error/Epochs ' + hyperparameters["optAlg"])
# plt.ylabel('Error')
# plt.xlabel('Epochs')
# plt.axis([0, numEpochs * numTrainBatches, min(errorDict["trainError"]), max(errorDict["trainError"])])
# plt.plot(errorDict["epoch"], errorDict["trainError"], 'r-', errorDict["epoch"], errorDict["valError"], 'b-', label = 'errors')

# fig = plt.figure(4)
# fig.canvas.set_window_title('AUC/Epochs')
# plt.title('AUC/Epochs ' + hyperparameters["optAlg"])
# plt.ylabel('AUC')
# plt.xlabel('Epochs')
# plt.axis([0, numEpochs * numTrainBatches, min(AUCDict["trainAUC"]), max(AUCDict["trainAUC"])])
# plt.plot(AUCDict["epoch"], AUCDict["trainAUC"], 'r-', AUCDict["epoch"], AUCDict["valAUC"], 'b-', label = 'AUCs')

# plt.show()

# reported AUCval for adaDelta = 0.480109327195
# reported AUCval for RMSProp = 0.825293957743