from src.helpFunctions import loadTrainingData, generateSubmission, initalizeModel, trainWithEarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import theano
folder = "saves\\16L500LS0.85FD0.9D0SPD1ISO1IPS1IP5.00C0.00L10.50L20.75RHO1e-10EPS13002242\\"
hyperparameters = {'c': 5, 'SPDhits': 0.49, 'dropout': 0.9, 'firstLayerDropout': 0.85, 
'layerSize': 500, 'rho': 0.75, 'epsilon': 1e-10, 'numLayers': 16, 'lambda1': 1e-4, 'lambda2': 0.5}

hyperparameters["isolationc"] = 1
hyperparameters["IPSig"] = 1
hyperparameters["IP"] = 1

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

Xtrain, Ytrain, Xval, Yval, mu, sig, AUCindices = loadTrainingData(variables, rng, valPart = 0.3, verbose = True)
model = initalizeModel(**hyperparameters)

model.loadModel("", folder, verbose = True)

generateSubmission(model, variables, mu, sig)