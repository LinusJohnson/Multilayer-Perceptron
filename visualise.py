# plot error over number of iterations
from matplotlib.pylab import matshow
plt.close('all')
%matplotlib qt
rng = np.random.RandomState(123)
numLayers = 5
Dropout = 0.7
layerSize = 50
firstLayerDropout = 0.8
SPDhits = 0
Lambda1 = 0
Lambda2 = 0
c = 3.5
batchSize = 300
variables = ['FlightDistance','FlightDistanceError', 'LifeTime', 'VertexChi2','pt','dira','DOCAone',
             'DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2', 'isolationa', 'isolationb', 'isolationd',
             'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 
             'p2_IsoBDT', 'p0_track_Chi2Dof','p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_pt','p0_p','p0_eta',
             'p0_IP','p0_IPSig','p1_pt','p1_p', 'p1_eta','p1_IP','p1_IPSig','p2_pt','p2_p','p2_eta','p2_IP',
             'p2_IPSig', 'IP', 'IPSig', 'isolationc']
Xtrain, Ytrain, Xval, Yval, mu,sig, AUCindices = loadTrainingData(variables, rng, verbose = True)
numTrainBatches = Xtrain.get_value(borrow=True).shape[0] / batchSize
numValidBatches = Xval.get_value(borrow=True).shape[0] / batchSize
numEpochs = 1000
trainErrors = []
valErrors = []
plt.figure(1)
plt.title('Error/Iterations')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.axis([0, numEpochs * numTrainBatches, 0, 1.5])
plt.figure(2)
plt.title('AUCs')
plt.ylabel('AUC score')
plt.xlabel('Iterations')
plt.axis([0, numEpochs * numTrainBatches, 0.8, 1])
plt.ion()
plt.show()
optAlg = ["RMSProp", "vSGDfd"]
mlp = initalizeModel(rng, numLayers, layerSize, firstLayerDropout, Dropout, Lambda1, Lambda2, c, len(variables), optAlg = optAlg[1])
trainFunction = mlp.train(Xtrain, Ytrain, batchSize)
validationErrorFunction = mlp.validationError(Xval, Yval, batchSize)
print "AUCtrain for random:", AUC(mlp, Xtrain, Ytrain, AUCindices["train"])
print "AUCval for random:", AUC(mlp, Xval, Yval, AUCindices["val"])
if mlp.optAlg == "vSGDfd":
    print "Slow starting vSGDfd..."
    for i in xrange(int(np.clip(0.001 * Xtrain.get_value(borrow=True).shape[0] / batchSize, 1, numTrainBatches))):
    #for i in xrange(5):
        cost = trainFunction(i, 1)
        #print np.asarray(model.thetas[0].eval())
    print "Done with slow start"
for epoch in xrange(numEpochs):
    for minibatchIndex in xrange(numTrainBatches):
        iter = epoch * numTrainBatches + minibatchIndex
        #print "alpha1:", np.asarray(mlp.alphas[0].eval())
        #print "alpha2:", np.asarray(mlp.alphas[1].eval())
        #print "theta1:", np.asarray(mlp.thetas[0].eval())
        #gradDebug = theano.function(inputs = [], 
        #                outputs = T.grad(cost = mlp.binaryCrossEntropyCostFunction(mlp.forwardProp(Xtrain), Ytrain, lambda1, lambda1, trainingSetSize2), wrt = mlp.alphas, disconnected_inputs = 'raise'), 
        #                name="gradDebug",
        #                givens = {X: Xtrain[:index,:], 
        #                          Y: Ytrain[:index,:]},
        #                allow_input_downcast = True)
        #                #,mode = theano.compile.MonitorMode(pre_func = inspect_inputs, post_func = inspect_outputs))
        #                #,mode = theano.compile.MonitorMode(post_func = detect_nan))
        #print "grad:", gradDebug()
        if mlp.optAlg == "vSGDfd": 
            cost = trainFunction(minibatchIndex, 0)
        else:
            cost = trainFunction(minibatchIndex)
        if ((iter + 1) % 100 == 0 or iter == 0):
            trainError = cost
            valError = np.mean([validationErrorFunction(i) for i in xrange(numValidBatches)])
            AUCtrain = AUC(mlp, Xtrain, Ytrain, AUCindices["train"])
            AUCval = AUC(mlp, Xval, Yval, AUCindices["val"])

            #trainErrors.append(np.asarray(trainError))
            #valErrors.append(np.asarray(valError))
            print "Iteration:", iter + 1, ", epoch", epoch, ", minibatchIndex", minibatchIndex, ", Error:", cost, ", validation error:", valError
            #print "Correlation:", cvm, cvm < 0.002, ", Agreement:", ks, ks < 0.09,", AUC:", AUC, ", AUCVal:", AUCval
            #for layer in mlp.layers:
            #    print np.asarray(layer.mean.eval()), np.asarray(layer.var.eval())
            #print "lastAlphaDelta:", mlp.lastAlphaDeltas[5]
            plt.figure(1)
            plt.scatter(iter, trainError, c = 'r', marker = 'x', label = 'trainError')
            plt.scatter(iter, valError, c = 'b', marker = 'o', label = 'valError')
            #plt.scatter(i, ks, c = 'g', marker = '+', label = 'ks')
            #plt.scatter(i, cvm, c = 'c', marker = '*', label = 'cvm')
            plt.figure(2)
            plt.scatter(iter, AUCtrain, c = 'r', marker = 'x', label = 'AUC')
            plt.scatter(iter, AUCval, c = 'b', marker = 'o', label = 'AUCval')
            fignum = 3
            fig = plt.figure(fignum)
            fig.clear()
            fig.canvas.set_window_title('alphas')
            for i in range(len(mlp.alphas)):
                ax = fig.add_subplot(1, len(mlp.alphas), i)
                matplot = ax.matshow(np.asarray(mlp.alphas[i].T.eval()), cmap=plt.get_cmap("Blues"))
                fig.colorbar(matplot)
            fignum += 1
            fig = plt.figure(fignum)
            fig.clear()
            fig.canvas.set_window_title('betas')
            for i in range(len(mlp.betas)):
                ax = fig.add_subplot(1, len(mlp.betas), i)
                matplot = ax.matshow(np.asmatrix(mlp.betas[i].T.eval()), cmap=plt.get_cmap("Greens"))
                fig.colorbar(matplot)
            fignum += 1
            fig = plt.figure(fignum)
            fig.clear()
            fig.canvas.set_window_title('gammas')
            for i in range(len(mlp.gammas)):
                ax = fig.add_subplot(1, len(mlp.gammas), i)
                matplot = ax.matshow(np.asmatrix(mlp.gammas[i].T.eval()), cmap=plt.get_cmap("Purples"))
                fig.colorbar(matplot)
            fignum += 1
            for i in range(len(mlp.thetas)):
                fig = plt.figure(i + fignum)
                fig.clear()
                plt.title('Theta for layer ' + str(i))
                fig.canvas.set_window_title('Theta for layer ' + str(i))
                ax = fig.add_subplot(111)
                matplot = ax.matshow(np.asarray(mlp.thetas[i].eval()), cmap=plt.get_cmap("Reds"))
                fig.colorbar(matplot)
            fignum += len(mlp.thetas)
            plt.draw()
            plt.pause(0.0001)