import theano
from theano import tensor as T
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcess
from scipy.stats import norm
from pandas import read_csv
import matplotlib.pyplot as plt
import time
from src.helpFunctions import loadTrainingData, initalizeModel, trainWithEarlyStopping, featureNormalization
class BayesianOptimization(object):
    def __init__(self, bounds, rng, optAlg, batchSize, kFold, stdfunction = None, numEpochs = 1, validationFrequency = 1, patience = 1, visualize = False, test = False):
        if stdfunction:
            self.stdfunction = stdfunction
        else:
            self.stdfunction = None
        self.bounds = bounds
        self.keys = list(bounds.keys())
        self.dim = len(bounds)
        self.boundsValues = []
        for key in self.bounds.keys():
            self.boundsValues.append(self.bounds[key])
        self.boundsValues = np.asarray(self.boundsValues)
        self.initPoints = []
        self.x_init = []
        self.y_init = []
        self.numEpochs = numEpochs
        self.validationFrequency = validationFrequency
        self.totalDuration = 0
        self.rng = rng
        self.test = test
        self.optAlg = optAlg
        self.batchSize = batchSize
        self.kFold = kFold
        self.patience = patience
        self.visualize = visualize
        if self.visualize:
            for i in xrange(len(self.keys)):
                paramName = self.keys[i]
                fig = plt.figure(i)
                fig.canvas.set_window_title(paramName)
                plt.title(paramName)
                plt.ylabel('Score')
                plt.xlabel(paramName)
                plt.axis([self.bounds[paramName][0] - abs(self.bounds[paramName][0]) * 0.1, self.bounds[paramName][1] + abs(self.bounds[paramName][0]) * 0.1, -1, 1.5])
        if visualize: 
                plt.ion()
                plt.show()
        # so that the same data split is used all the time otherwise there will be randomness in the evaluation
        if stdfunction is None:
            data = read_csv('tau_data/' + 'training.csv', index_col='id')
            print "Shuffling data..."
            self.shuffledData = data.reindex(rng.permutation(data.index))
        self.timestamp = time.strftime("%d-%m-%Y-%Hh%Mm%Ss")
        if test:
            self.subfolder = "BayOptSaves\\" + "TEST" + str(self.timestamp) + "\\"
        else:
            self.subfolder = "BayOptSaves\\" + str(self.timestamp) + "\\"
        self.logfile = open("BayesianOptimizationLogfiles\\" + self.timestamp + ".txt",'w')
        self.logfile.write('Bayesian Optimization\n')

    def init(self, numInitPoints, numIter):
        print "Initalizing..."
        self.logfile.write("Initalizing...\n")
        randX = [self.rng.uniform(low = x[0], high = x[1], size = numInitPoints) for x in self.boundsValues]
        self.initPoints += list(map(list, zip(*randX)))
        completedInitPoints = []
        y_init = []
        i = 0.0
        for x in self.initPoints:
            i += 1.0
            t0 = time.time()
            if self.stdfunction:
                print dict(zip(self.keys, x))
                y_init.append(self.stdfunction(**dict(zip(self.keys, x))))
            else:
                y_init.append(self.function(x))
            if self.visualize: self.plot(dict(zip(self.keys, x)), y_init[-1])
            print "test:", {y_init[i]: dict(zip(self.keys, self.initPoints[i])) for i in xrange(len(y_init))}
            self.logfile.write("Evaluated models: " + str({y_init[i]: dict(zip(self.keys, self.initPoints[i])) for i in xrange(len(y_init))}))
            self.totalDuration += time.time() - t0
            avTime = (self.totalDuration / i) / 60.0
            print "Average duration: ", avTime, "minutes"
            estTotTime = avTime * (len(self.initPoints) + numIter)
            print "Estimated total time:", estTotTime, "minutes"
            estRemTime = avTime * (len(self.initPoints) + numIter - i)
            print "Estimated remaining time:", estRemTime, "minutes"
            Progress = (1. - estRemTime / (estTotTime + 1e-9)) * 100.
            print "Progress:", Progress, "%"
            self.logfile.write("Average duration: " + str(avTime) + " minutes" + "\n" \
                               + "Estimated total time: " + str(estTotTime) + " minutes" + "\n" \
                               + "Estimated remaining time: " + str(estRemTime) + " minutes" + "\n" \
                               + "Progress: " + str(Progress) + " %" + "\n")
        self.initPoints += self.x_init
        y_init += self.y_init
        self.X = np.asarray(self.initPoints)
        self.Y = np.asarray(y_init)
        
    def initialize(self, X_dicts, Y):
        for i in xrange(len(Y)):
            self.y_init.append(Y[i])
            all_points = []
            for key in self.keys: all_points.append(X_dicts[i][key])
            self.x_init.append(all_points)
    
    def explore(self, points_dict):
        all_points = []
        for key in self.keys: all_points.append(points_dict[key])
        self.initPoints = list(map(list, zip(*all_points)))

    def uniqueRows(self, a):
        order = np.lexsort(a.T)
        reorder = np.argsort(order)
        a = a[order]
        diff = np.diff(a, axis=0)
        ui = np.ones(len(a), 'bool')
        ui[1:] = (diff != 0).any(axis=1)
        return ui[reorder]
    
    def plot(self, x, y):
        for i in xrange(len(self.keys)):
            fig = plt.figure(i)
            plt.scatter(x[self.keys[i]], y, c = 'r', marker = 'x')
        plt.draw()
        plt.pause(0.0001)

    def AcquisitionFunction(self, x, gp, ymin):
        mean, var = gp.predict(x, eval_MSE=True)
        if var == 0:
            return 0
        else:
            Z = (mean - ymin) / (np.sqrt(var) + 1e-9)
            return (mean - ymin) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)

    def minimize(self, numInitPoints = 5, restarts = 50, numIter = 25):
        self.init(numInitPoints, numIter)
        gp = GaussianProcess(random_start = 25, random_state = self.rng, verbose = False)
        ur = self.uniqueRows(self.X)
        gp.fit(self.X[ur], self.Y[ur])
        ymin = self.Y.min()
        x_min = self.minimizeAcquisitionFunction(gp, ymin, restarts, self.boundsValues)
        print "Minimizing..."
        self.logfile.write("Minimizing...\n")
        for i in xrange(numIter):
            t0 = time.time()
            print "BayOp Iteration:", i + 1
            self.logfile.write("BayOp Iteration:" + str(i) + '\n')
            self.X = np.concatenate((self.X, x_min.reshape((1, self.dim))), axis=0)
            if self.stdfunction:
                self.Y = np.append(self.Y, self.stdfunction(**dict(zip(self.keys, x_min))))
            else:
                self.Y = np.append(self.Y, self.function(x_min))
            if self.visualize: self.plot(dict(zip(self.keys, self.X[-1])), self.Y[-1])
            self.logfile.write("Evaluated models: " + str({self.Y[i]: dict(zip(self.keys, self.X[i])) for i in xrange(len(self.Y))}))
            ur = self.uniqueRows(self.X)
            gp.fit(self.X[ur], self.Y[ur])
            if self.Y[-1] < ymin:
                ymin = self.Y[-1]
            x_min = self.minimizeAcquisitionFunction(gp, ymin, restarts, self.boundsValues)
            self.totalDuration += time.time() - t0
            avTime = (self.totalDuration / (len(self.initPoints) + (i + 1.))) / 60.0
            print "Average duration: ", avTime, "minutes"
            estTotTime = avTime * (len(self.initPoints) + numIter)
            print "Estimated total time:", estTotTime, "minutes"
            estRemTime = avTime * (numIter - (i + 1.))
            print "Estimated remaining time:", estRemTime, "minutes"
            Progress = (1. - estRemTime / (estTotTime + 1e-9)) * 100
            print "Progress:", Progress, "%"
            self.logfile.write("Average duration: " + str(avTime) + " minutes" + "\n" \
                               + "Estimated total time: " + str(estTotTime) + " minutes" + "\n" \
                               + "Estimated remaining time: " + str(estRemTime) + " minutes" + "\n" \
                               + "Progress: " + str(Progress) + " %" + "\n")        
        self.res = {}
        self.res['min'] = {'min_val': self.Y.min(), 'min_params': dict(zip(self.keys, self.X[self.Y.argmin()]))}
        self.res['all'] = {'values': [], 'params': []}
        for t, p in zip(self.Y, self.X):
            self.res['all']['values'].append(t)
            self.res['all']['params'].append(dict(zip(self.keys, p)))
        print('Optimization finished with minimum: %8f, at position: %8s.' % (self.res['min']['min_val'], self.res['min']['min_params']))
        self.logfile.write(str(self.res) + "\n")
        self.logfile.write(str('Optimization finished with minimum: %8f, at position: %8s.' % (self.res['min']['min_val'], self.res['min']['min_params'])) + "\n")
        self.logfile.close()

    def minimizeAcquisitionFunction(self, gp, ymin, restarts, bounds):
        x_min = bounds[:, 0]
        ei_min = np.inf
        for i in range(restarts):
            x_try = np.asarray([self.rng.uniform(x[0], x[1], size=1) for x in bounds]).T
            res = minimize(lambda x: self.AcquisitionFunction(x, gp = gp, ymin = ymin), x_try, bounds = bounds, method = 'L-BFGS-B')
            if res.fun <= ei_min:
                x_min = res.x
                ei_min = res.fun
        return x_min

    def function(self, x):
        xdict = dict(zip(self.keys, x))
        print ""
        print xdict
        if self.test:
            modelName = "TEST"
        else:
            modelName = ""
            if 'numLayers' in xdict: modelName += "%iL" % int(xdict['numLayers'])
            if 'layerSize' in xdict: modelName += "%iLS" % int(xdict['layerSize'])
            if 'firstLayerDropout' in xdict: modelName += "%.2fFD" % xdict['firstLayerDropout']
            if 'dropout' in xdict: modelName += "%gD" % xdict['dropout']
            if 'SPDhits' in xdict: modelName += "%iSPD" % np.round(xdict['SPDhits'])
            if 'isolationc' in xdict: modelName += "%iISO" % np.round(xdict['isolationc'])
            if 'IPSig' in xdict: modelName += "%iIPS" % np.round(xdict['IPSig'])
            if 'IP' in xdict: modelName += "%iIP" % np.round(xdict['IP'])
            if 'c' in xdict: modelName += "%.2fC" % xdict['c']
            if 'lambda1' in xdict: modelName += "%.2fL1" % xdict['lambda1']
            if 'lambda2' in xdict: modelName += "%.2fL2" % xdict['lambda2']
            if 'rho' in xdict: modelName += "%.2fRHO" % xdict['rho']
            if 'epsilon' in xdict: modelName += "%gEPS" % xdict['epsilon']
            if 'learningRate' in xdict: modelName += "%gLR" % xdict['learningRate']
        self.logfile.write("Model name " + modelName + "\n")
        self.logfile.write(str(xdict) + "\n")
        variables = ['FlightDistance','FlightDistanceError', 'LifeTime', 'VertexChi2','pt','dira','DOCAone',
                     'DOCAtwo','DOCAthree','IP_p0p2','IP_p1p2', 'isolationa', 'isolationb', 'isolationd',
                     'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3', 'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 
                     'p2_IsoBDT', 'p0_track_Chi2Dof','p1_track_Chi2Dof', 'p2_track_Chi2Dof','p0_pt','p0_p','p0_eta',
                     'p0_IP','p0_IPSig','p1_pt','p1_p', 'p1_eta','p1_IP','p1_IPSig','p2_pt','p2_p','p2_eta','p2_IP',
                     'p2_IPSig']
        if 'SPDhits' in xdict: 
            if np.round(xdict.pop('SPDhits')): variables += ['SPDhits']
        else: variables += ['SPDhits']
        if 'isolationc' in xdict:
            if np.round(xdict.pop('isolationc')): variables += ['isolationc']
        else: variables += ['isolationc']
        if 'IPSig' in xdict:
            if np.round(xdict.pop('IPSig')): variables += ['IPSig']
        else: variables += ['IPSig']
        if 'IP' in xdict:
            if np.round(xdict.pop('IP')): variables += ['IP']
        else: variables += ['IP']
        xdict['numFeatures'] = len(variables)
        xdict['logfile'] = self.logfile
        xdict['rng'] = self.rng
        xdict['optAlg'] = self.optAlg
        kFoldData, kFoldAUCindices = loadTrainingData(variables, self.rng, shuffledData = self.shuffledData, kFold = self.kFold, verbose = True)
        scores = [0] * self.kFold
        for i in xrange(self.kFold):
            mlp = initalizeModel(**xdict)
            print "k-fold:", i + 1
            self.logfile.write("k-fold: " + str(i + 1) + "\n")
            XtrainData = np.concatenate(kFoldData[:i] + kFoldData[i+1:], axis=0)[:,1:]
            YtrainData = np.concatenate(kFoldData[:i] + kFoldData[i+1:], axis=0)[:,0]
            XvalData = kFoldData[i][:,1:]
            YvalData = kFoldData[i][:,0]

            # feature normalization
            XtrainData, mu, sig = featureNormalization(XtrainData)
            XvalData = featureNormalization(XvalData, mu, sig)

            AUCindices = {"train": kFoldAUCindices[:i] + kFoldAUCindices[i+1:], "val": kFoldAUCindices[i]}
            
            Xtrain = theano.shared(value = np.matrix(XtrainData).astype(theano.config.floatX), name = "Xtrain", borrow = True)
            Ytrain = theano.shared(value = np.matrix(YtrainData).T.astype(theano.config.floatX), name = "Ytrain", borrow = True)
            Xval = theano.shared(value = np.matrix(XvalData).astype(theano.config.floatX), name = "Xval", borrow = True)
            Yval = theano.shared(value = np.matrix(YvalData).T.astype(theano.config.floatX), name = "Yval", borrow = True)
            
            data = Xtrain, Ytrain, Xval, Yval, mu, sig, AUCindices
            scores[i]= trainWithEarlyStopping(mlp, self.numEpochs, self.patience, self.validationFrequency, data, variables, self.batchSize, logfile = self.logfile, name = modelName, verbose = True, save = True, subfolder = self.subfolder)
            print "Score:", scores[i]
            self.logfile.write("Score: " + str(scores[i]) + "\n")
            del data, Xtrain, Ytrain, XtrainData, YtrainData, XvalData, YvalData, Xval, Yval, mu, sig, AUCindices, mlp
        score = np.mean(scores)
        print "Scores:", scores
        print "Average k-fold score:", score
        self.logfile.write("Scores: " + str(scores) + "\n")
        self.logfile.write("Average k-fold score: " + str(score) + "\n")
        del kFoldData, kFoldAUCindices, scores
        return score