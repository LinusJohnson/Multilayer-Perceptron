import theano
from theano import tensor as T
import evaluation
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from collections import deque
import time
import os
from src.mlp import MLP

# from src.helpFunctions import loadTrainingData, featureNormalization, agreement, correlation, AUC, generateSubmission, initalizeModel, trainWithEarlyStopping

def loadTrainingData(variables, rng, valPart = 0.3, shuffledData = None, kFold = None, verbose = True):
    if shuffledData is None:
        data = read_csv('E:\\FlavoursOfPhysics\\flavours-of-physics-start-master\\tau_data\\' + 'training.csv', index_col='id')
        if verbose: print "Shuffling data..."
        data = data.reindex(rng.permutation(data.index))
    else:
        data = shuffledData

    dataSize = data.shape[0]
    
    if kFold is None:
        if verbose: print "Selecting data for the computation of the weighted AUC..."
        split = dataSize*(1-valPart)
        AUCindicesTemp = np.asarray(data['min_ANNmuon'] > 0.4)
        AUCindices = {"train": AUCindicesTemp[:split], "val": AUCindicesTemp[split:]}
        data = data[['signal'] + variables].values
        dataXtrain = data[:split,1:]
        dataYtrain = data[:split,0]
        dataXval = data[split:,1:]
        dataYval = data[split:,0]

        posTrain = np.sum(dataYtrain == 1)
        negTrain = np.sum(dataYtrain == 0)
        print "number of positive labels:", posTrain
        print "number of negative labels:", negTrain
        print "ratio pos/neg:", posTrain / (1.*negTrain)
        print "%%positive:", posTrain / (1.*negTrain + posTrain)
        print "%%negative:", negTrain / (1.*negTrain + posTrain)
        posVal = np.sum(dataYval == 1)
        negVal = np.sum(dataYval == 0)
        print "number of positive labels:", posVal
        print "number of negative labels:", negVal
        print "ratio pos/neg:", posVal / (1.*negVal)
        print "%%positive:", posVal / (1.*negVal + posVal)
        print "%%negative:", negVal / (1.*negVal + posVal)

        # feature normlization
        dataXtrain, mu, sig = featureNormalization(dataXtrain)
        dataXval = featureNormalization(dataXval, mu, sig)

        Xtrain = theano.shared(value = np.matrix(dataXtrain).astype(theano.config.floatX), name = "Xtrain", borrow = True)
        Ytrain = theano.shared(value = np.matrix(dataYtrain).T.astype(theano.config.floatX), name = "Ytrain", borrow = True)
        Xval = theano.shared(value = np.matrix(dataXval).astype(theano.config.floatX), name = "Xval", borrow = True)
        Yval = theano.shared(value = np.matrix(dataYval).T.astype(theano.config.floatX), name = "Yval", borrow = True)

        return Xtrain, Ytrain, Xval, Yval, mu, sig, AUCindices
    else:
        kFoldData = np.array_split(data[['signal'] + variables].values, kFold, axis=0)
        kFoldAUCindices = np.array_split(np.asarray(data['min_ANNmuon'] > 0.4), kFold, axis=0)
        return kFoldData, kFoldAUCindices
        
def featureNormalization(data, mu = None, sig = None):
    if mu is None:
        mu = np.mean(data, axis = 0)
        sig = np.std(data, axis = 0)
        return (data - mu) / (sig + 1e-9), mu, sig
    else:
        return (data - mu) / (sig + 1e-9)

def agreement(model, variables, mu, sig):
    check_agreement = read_csv('E:\\FlavoursOfPhysics\\flavours-of-physics-start-master\\tau_data\\' + 'check_agreement.csv', index_col='id')
    predictionFunction = model.predict()
    split = 6
    splitPredictionData = np.array_split(featureNormalization(check_agreement[variables].values, mu, sig), split, axis=0)
    agreement_probs = np.asarray(predictionFunction(splitPredictionData[0]))
    for i in xrange(1, split):
        agreement_probs = np.append(agreement_probs, np.asarray(predictionFunction(splitPredictionData[i])), axis = 0)
    return evaluation.compute_ks(agreement_probs[check_agreement['signal'].values == 0],
                                 agreement_probs[check_agreement['signal'].values == 1],
                                 check_agreement[check_agreement['signal'] == 0]['weight'].values,
                                 check_agreement[check_agreement['signal'] == 1]['weight'].values)

def correlation(model, variables, mu, sig):
    check_correlation = read_csv('E:\\FlavoursOfPhysics\\flavours-of-physics-start-master\\tau_data\\' + 'check_correlation.csv', index_col='id')
    correlation_probs = np.asarray(model.predict()(np.matrix(featureNormalization(check_correlation[variables].values, mu, sig)).astype(theano.config.floatX))).T[0]
    return evaluation.compute_cvm(correlation_probs, check_correlation['mass'])

def AUC(model, dataX, dataY, AUCindex):
    return evaluation.roc_auc_truncated(np.asarray(dataY.eval())[AUCindex], np.asarray(model.predict()(np.asarray(dataX.eval())[AUCindex])).T[0])

def generateSubmission(model, variables, mu, sig):
    predict = model.predict()
    test = read_csv('E:\\FlavoursOfPhysics\\flavours-of-physics-start-master\\tau_data\\' + 'test.csv', index_col='id')
    testDataX = featureNormalization(test[variables].values, mu, sig)
    result1 = DataFrame({'id': test.index[:855819 / 3.0]})
    result1['prediction'] = np.asarray(predict(testDataX[:855819 / 3.0, :])).T[0]
    result2 = DataFrame({'id': test.index[855819 / 3.0 : 855819 * (2.0 / 3.0)]})
    result2['prediction'] = np.asarray(predict(testDataX[855819 / 3.0 : 855819 * (2.0 / 3.0), :])).T[0]
    result3 = DataFrame({'id': test.index[855819 * (2.0 / 3.0) :]})
    result3['prediction'] = np.asarray(predict(testDataX[855819 * (2.0 / 3.0) :, :])).T[0]
    result = result1.append([result2, result3])
    result.to_csv('submission.csv', index = False, sep = ',')
    print "Done"

def initalizeModel(rng, numLayers, layerSize, firstLayerDropout, dropout, lambda1, lambda2, c, numFeatures, optAlg, rho = 0.95, epsilon = 1e-9, learningRate = 0.001, momentum = 0.9, logfile = None, verbose = True):
    numLayers = int(numLayers)
    functions = [T.nnet.sigmoid, T.nnet.ultra_fast_sigmoid, T.nnet.hard_sigmoid, T.nnet.relu, T.nnet.softmax, T.nnet.softplus, T.tanh]
    probabilities = [firstLayerDropout] + [dropout for i in xrange(numLayers)]
    layerSizes = [numFeatures] + [max(int(layerSize), 1) for i in xrange(numLayers)] + [1.0]
    activations = [functions[3] for i in xrange(numLayers)] + [functions[0]]
    if verbose:
        print "Number of layers:", numLayers
        print "Layer sizes:", map(lambda x: int(x), layerSizes)
        print "Layer activation functions:", map(lambda x: x.name if hasattr(x, 'name') else x.__name__, activations)
        print "Regularisation terms: lambda 1 (L1) =", lambda1, ", lambda 2 (L2) =", lambda2, ", max-norm constraint constant:", c
        print "Using", optAlg, "as optimization algorithm"
    if logfile:
        logfile.write("Number of layers: " + str(numLayers) + "\n" + 
         "Layer sizes: " + str(map(lambda x: int(x), layerSizes)) + "\n" +
         "Layer activation functions: " + str(map(lambda x: x.name if hasattr(x, 'name') else x.__name__, activations)) + "\n" +
         "Regularisation terms: lambda 1 (L1) = " + str(lambda1) + ", lambda 2 (L2) = " + str(lambda2) + ", max-norm constraint constant: " + str(c) + "\n" +
         "Using " + optAlg + " as optimization algorithm" + "\n")
    mlp = MLP(rng = rng, activations = activations, layerSizes = layerSizes, probabilities = probabilities, lambda1 = lambda1, lambda2 = lambda2, c = c, optAlg = optAlg, learningRate = np.float32(learningRate), rho = np.float32(rho), epsilon = np.float32(epsilon), momentum = np.float32(momentum))
    return mlp

def trainWithEarlyStopping(model, numEpochs, patience, validationFrequency, data, variables, batchSize, errorDict = None, AUCDict = None, logfile = None, name = "", debug = False, verbose = True, save = True, subfolder = "", visualize = False):
    Xtrain, Ytrain, Xval, Yval, mu, sig, AUCindices = data
    numTrainBatches = Xtrain.get_value(borrow=True).shape[0] / batchSize
    numValidBatches = Xval.get_value(borrow=True).shape[0] / batchSize
    patienceIncrease = 1.75
    improvementThreshold = 0.999
    stoppingCriteria = 10
    bestValError = np.inf
    bestvalIter = 0
    bestAUCval = 0
    skipAUCEval = 0 #800
    endIteration = False
    t0 = time.time()
    trainFunction = model.train(Xtrain, Ytrain, batchSize)
    validationErrorFunction = model.validationError(Xval, Yval)
    import theano.d3viz as d3v
    print "Building graph..."
    d3v.d3viz(trainFunction, 'd3viz/mlp.html')
    if verbose: print "Compilation time:", (time.time() - t0) / 60.0, "minutes"
    if logfile: logfile.write("Compilation time: " + str((time.time() - t0) / 60.0) + " minutes" + "\n")
    generalizationLoss = 0
    trainingProgress = 0
    trainingStrip = deque([0.0 for i in xrange(max(validationFrequency, 20))])
    timestamp = time.strftime("%d%H%M%S")
    folder = ""
    printFrequency = 10
    iter = 0
    if visualize: import matplotlib.pyplot as plt
    if errorDict is not None:
        if visualize:
            fig = plt.figure(1)
            fig.canvas.set_window_title('Error/Epochs')
            plt.title('Error/Epochs ' + model.optAlg)
            plt.ylabel('Error')
            plt.xlabel('Epochs')
            plt.axis([0, numEpochs, 0, 50000])
        errorDict["epoch"] = []
        errorDict["trainError"] = []
        errorDict["valError"] = []
    if AUCDict is not None:
        if visualize:
            fig = plt.figure(2)
            fig.canvas.set_window_title('AUC/Epochs')
            plt.title('AUC/Epochs ' + model.optAlg)
            plt.ylabel('AUC')
            plt.xlabel('Epochs')
            plt.axis([0, numEpochs, 0.5, 1])
        AUCDict["epoch"] = []
        AUCDict["trainAUC"] = []
        AUCDict["valAUC"] = []
    if visualize:
        plt.ion()
        plt.show()
    if save:
        folder = 'saves\\' + subfolder + name + timestamp + '\\'
        if logfile: logfile.write("Folder to save in " + folder + "\n")
        os.makedirs(folder)
    if model.optAlg == "vSGDfd":
        if verbose: print "Slow starting vSGDfd..."
        for i in xrange(int(np.clip(0.001 * Xtrain.get_value(borrow=True).shape[0] / batchSize, 1, numTrainBatches))):
            cost = trainFunction(i, 1)
        if verbose: print "Done with slow start"
    t0 = time.time()
    if verbose: print "Training..."
    if logfile: logfile.write("Training..." + "\n")
    for epoch in xrange(numEpochs):
        for minibatchIndex in xrange(numTrainBatches):
            if model.optAlg == "vSGDfd": 
                cost = trainFunction(minibatchIndex, 0)
            else:
                cost = trainFunction(minibatchIndex)
            trainingStrip.pop()
            trainingStrip.appendleft(cost)
            iter = epoch * numTrainBatches + minibatchIndex
            if (iter + 1) % validationFrequency == 0:
                #valError = np.mean([validationErrorFunction(i) for i in xrange(numValidBatches)])
                valError = validationErrorFunction()
                if logfile: logfile.write("Iteration: " + str(iter + 1) + ", epoch: " + str(epoch) + ", minibatchIndex: " + str(minibatchIndex) +", validation frequency: " + str(validationFrequency) + ", Error: " + str(cost) + ", validation error: " + str(valError) + "\n")
                if valError < bestValError:
                    if verbose: print "Iteration:", iter + 1, ", epoch", epoch, ", minibatchIndex", minibatchIndex, ", validation frequency:", validationFrequency, ", Error:", cost, ", validation error:", valError
                    if valError < bestValError * improvementThreshold:
                        newPatience = max(patience, int(iter * patienceIncrease))
                        if newPatience > patience:
                            patience = newPatience
                            if verbose: print "valError has decreased significantly so the patience is updated to:", patience
                            if logfile: logfile.write("valError has decreased significantly so the patience is updated to: " + str(patience) + "\n")
                    if verbose: print "Best valError updated to:" , valError
                    if logfile: logfile.write("Best valError updated to: " + str(valError) + "\n")
                    AUCval = 0
                    if iter > skipAUCEval: AUCval = AUC(model, Xval, Yval, AUCindices["val"])
                    if verbose: print "current AUCval:", AUCval, ", bestAUCval:", bestAUCval
                    if logfile: logfile.write("current AUCval: " + str(AUCval) + ", bestAUCval: " + str(bestAUCval) + "\n")
                    if save and AUCval > bestAUCval:
                        if verbose: print "AUC has improved. Saving..."
                        if logfile: logfile.write("AUC has improved. Saving..." + "\n")
                        model.saveModel(name = "", folder = folder, verbose = False)
                        bestAUCval = AUCval
                    bestvalIter = iter
                    bestValError = valError
                # to limit the amount of printing to console
                elif (iter + 1) % (printFrequency * validationFrequency) == 0:
                    if verbose: print "Iteration:", iter + 1, ", epoch", epoch, ", minibatchIndex", minibatchIndex, ", validation frequency:", validationFrequency, ", Error:", cost, ", validation error:", valError
                if iter - bestvalIter > validationFrequency * 20 and validationFrequency != 1:
                    validationFrequency -= 1
                    bestvalIter = iter
                    if verbose: print "Validation error has not improved for 20 validations so validation frequency is updated to", validationFrequency
                    if logfile: logfile.write("Validation error has not improved for 20 validations so validation frequency is updated to " + str(validationFrequency) + "\n")
                if iter >= 20 and iter >= validationFrequency: # ensure the strip is filled
                    # from "Early Stopping - but when?" by Lutz Prechelt
                    generalizationLoss = 100 * (valError / (bestValError + 1e-9) - 1)
                    trainingProgress = 1000 * (sum(trainingStrip) / (len(trainingStrip) * min(trainingStrip) + 1e-9) - 1)
                    if generalizationLoss / (trainingProgress + 1e-9) > stoppingCriteria:
                            if verbose: print "PQ stopping criteria has been triggered. GL/P:", generalizationLoss / (trainingProgress + 1e-9)
                            if logfile: logfile.write("PQ stopping criteria has been triggered. GL/P: " + str(generalizationLoss / (trainingProgress + 1e-9)) + "\n")
                            endIteration = True
                            break
            if patience <= iter:
                if verbose: print "Patience exceeded"
                if logfile: logfile.write("Patience exceeded" + "\n")
                endIteration = True
                break
        if errorDict is not None:
            errorDict["epoch"].append(epoch)
            errorDict["trainError"].append(cost)
            errorDict["valError"].append(valError)
            if visualize:
                plt.figure(1)
                plt.plot(errorDict["epoch"], errorDict["trainError"], 'r-', errorDict["epoch"], errorDict["valError"], 'b-')
                plt.axis([0, max(1, errorDict["epoch"][-1]), min(errorDict["trainError"] + errorDict["valError"]), max(errorDict["trainError"] + errorDict["valError"])])
        if AUCDict is not None:
            AUCtrain = AUC(model, Xtrain, Ytrain, AUCindices["train"])
            AUCval = AUC(model, Xval, Yval, AUCindices["val"])
            AUCDict["epoch"].append(epoch)
            AUCDict["trainAUC"].append(AUCtrain)
            AUCDict["valAUC"].append(AUCval)
            if visualize:
                plt.figure(2)
                plt.plot(AUCDict["epoch"], AUCDict["trainAUC"], 'r-', AUCDict["epoch"], AUCDict["valAUC"], 'b-')
                plt.axis([0, max(1, AUCDict["epoch"][-1]), min(AUCDict["trainAUC"] + AUCDict["valAUC"]), max(AUCDict["trainAUC"] + AUCDict["valAUC"])])
        if visualize: 
            plt.draw()
            plt.pause(0.0001)
        if endIteration: break
    if verbose: print "Execution time:", (time.time() - t0) / 60.0, " minutes"
    if logfile: logfile.write("Execution time: " + str((time.time() - t0) / 60.0) + " minutes" + "\n")
    if save: model.loadModel(name = "", folder = folder, verbose = False)
    ks = agreement(model, variables, mu, sig)
    cvm = correlation(model, variables, mu, sig)
    AUCval = AUC(model, Xval, Yval, AUCindices["val"])
    if verbose: 
        print "Agreement: " + str(ks) + " under 0.09 " + str(ks < 0.09) + ", Correlation: " + str(cvm) + " under 0.002 " + str(cvm < 0.002)
        print "AUCval: " + str(AUCval)
    if logfile: 
        logfile.write("Agreement: " + str(ks) + " under 0.09 " + str(ks < 0.09) + ", Correlation: " + str(cvm) + " under 0.002 " + str(cvm < 0.002) + "\n")
        logfile.write("AUCval: " + str(AUCval) + "\n")
    return -AUCval + ((ks > 0.09) + (cvm > 0.002))