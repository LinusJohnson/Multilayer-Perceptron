import sys, getopt, time
def main(argv):
	verbose = False
	save = False
	logfile = None
	try:
		opts, args = getopt.getopt(argv,"v:s:l:",["verbose=","save=","logfile="])
	except getopt.GetoptError:
		print 'testMLP.py -v <verbose> -s <save>'
		sys.exit(2)
	for opt, arg in opts:
		if opt in ('-v', '--verbose'):
			verbose = arg.lower() in ("yes","true","t","1")
		elif opt in ('-s', '--save'):
			save = arg.lower() in ("yes","true","t","1")
		elif opt in ('-l', '--logfile'):
			if arg.lower() in ("yes","true","t","1"):
				logfile = open("TESTLogfiles\\" + time.strftime("%d-%m-%Y-%Hh%Mm%Ss") + ".txt", 'w')
				logfile.write('TEST logfile\n')
	if verbose and logfile is not None: print "Testing logging"
	elif verbose: print "Not testing logging"
	if verbose and save: print "Testing saving"
	elif verbose: print "Not testing saving"
	from src.helpFunctions import loadTrainingData, generateSubmission, initalizeModel, trainWithEarlyStopping
	import numpy as np
	#import theano
	#import os
	#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
	#theano.config.profile = True
	#theano.config.profile_memory = True
	numLayers = 1 #14
	Dropout = 0.5
	layerSize = 1 #200
	firstLayerDropout = 0.8
	Lambda1 = 0
	Lambda2 = 0
	c = 3
	learningRate = 0.0001
	rho = 0.95 # 0.95
	epsilon = 1e-6 # 1e-9
	rng = np.random.RandomState(123)
	variables = ['FlightDistance','FlightDistanceError', 'LifeTime', 'VertexChi2','pt','dira','DOCAone',
	             'DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2', 'isolationa', 'isolationb', 'isolationd',
	             'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 
	             'p2_IsoBDT', 'p0_track_Chi2Dof','p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_pt','p0_p','p0_eta',
	             'p0_IP','p0_IPSig','p1_pt','p1_p', 'p1_eta','p1_IP','p1_IPSig','p2_pt','p2_p','p2_eta','p2_IP',
	             'p2_IPSig', 'IP', 'IPSig', 'isolationc', 'SPDhits']
	data = loadTrainingData(variables, rng, valPart = 0.25, verbose = verbose)
	optAlgs = ["RMSProp", "adaDelta", "vSGDfd"]
	numEpochs = 3
	patience = 3
	validationFrequency = 1
	batchSize = 5000 #300

	optAlg = optAlgs[0]
	mlp = initalizeModel(rng, numLayers, layerSize, firstLayerDropout, Dropout, Lambda1, Lambda2, c, len(variables), optAlg, rho, epsilon, learningRate, verbose = verbose, logfile = logfile)
	trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, name = "TESTRMSProp", subfolder = "TESTs\\", verbose = verbose, save = save, logfile = logfile)

	optAlg = optAlgs[1]
	mlp = initalizeModel(rng, numLayers, layerSize, firstLayerDropout, Dropout, Lambda1, Lambda2, c, len(variables), optAlg, rho, epsilon, learningRate, verbose = verbose, logfile = logfile)
	trainWithEarlyStopping(mlp, numEpochs, patience, validationFrequency, data, variables, batchSize, name = "TESTadaDelta", subfolder = "TESTs\\", verbose = verbose, save = save, logfile = logfile)
	print "Test was successful"
if __name__ == "__main__":
   main(sys.argv[1:])