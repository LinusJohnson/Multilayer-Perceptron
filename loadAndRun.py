from src.helpFunctions import loadTrainingData, generateSubmission, initalizeModel, trainWithEarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import theano
folder = "saves\\21L500LS0.85FD0.9D0SPD1ISO1IPS1IP5.00C0.00L10.50L20.75RHO1e-10EPS12234434\\"
hyperparameters = {'c': 5, 'SPDhits': 0.49, 'dropout': 0.9, 'firstLayerDropout': 0.85, 
'layerSize': 500, 'rho': 0.75, 'epsilon': 1e-10, 'numLayers': 21, 'lambda1': 1e-4, 'lambda2': 0.5}

hyperparameters["isolationc"] = 1
hyperparameters["IPSig"] = 1
hyperparameters["IP"] = 1
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
rng = np.random.RandomState(123)
hyperparameters["rng"] = rng
variables = ['FlightDistance','FlightDistanceError', 'LifeTime', 'VertexChi2','pt','dira','DOCAone',
             'DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2', 'isolationa', 'isolationb', 'isolationd',
             'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 
             'p2_IsoBDT', 'p0_track_Chi2Dof','p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_pt','p0_p','p0_eta',
             'p0_IP','p0_IPSig','p1_pt','p1_p', 'p1_eta','p1_IP','p1_IPSig','p2_pt','p2_p','p2_eta','p2_IP',
             'p2_IPSig']
if np.round(hyperparameters.pop('SPDhits')): variables += ['SPDhits']
if np.round(hyperparameters.pop('isolationc')): variables += ['isolationc']
if np.round(hyperparameters.pop('IPSig')): variables += ['IPSig']
if np.round(hyperparameters.pop('IP')): variables += ['IP']
hyperparameters["numFeatures"] = len(variables)
optAlgs = ["RMSProp", "adaDelta", "vSGDfd"]
hyperparameters["optAlg"] = optAlgs[1]
mlp = initalizeModel(**hyperparameters)
mlp.loadModel("", folder, verbose = True)
data = loadTrainingData(variables, rng, valPart = 0.3, verbose = True)
optAlgs = ["RMSProp", "adaDelta", "vSGDfd"]
patience = 12000
validationFrequency = 1
batchSize = int(data[0].get_value(borrow=True).shape[0] / 6.) #5000 #300
numTrainBatches = data[0].get_value(borrow=True).shape[0] / batchSize
numEpochs = int(patience / numTrainBatches) * 15
trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, name = modelName)