import mltools as ml
import sys
import numpy as np
import matplotlib.pyplot as plt
from logisticClassify2 import *

iris = np.genfromtxt("data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1]  # get first two features & target
X,Y = ml.shuffleData(X,Y)  # reorder randomly (important later)
X,_ = ml.transforms.rescale(X)  # works much better on rescaled data
XA, YA = X[Y<2,:], Y[Y<2]  # get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0]  # get class 1 vs 2
'''plt.title('1.1 Scattered data points for XA, YA')
plt.scatter(XA[:,:1], YA, color = 'r')
plt.scatter(XA[:,1:], YA, color = 'b')
plt.figure()
plt.title('1.1 Scattered data points for XB, YB')
plt.scatter(XB[:,:1], YB, color = 'r')
plt.scatter(XB[:,1:], YB, color = 'b')'''
#1.1 done

learnerA = logisticClassify2() # create "blank"learner
learnerA.classes = np.unique(YA) # define class labels using YA or YB
wts = np.array([0.5,-0.25,1]) # TODO: fill in values
learnerA.theta = wts # set the learner’s parameters
'''plt.figure()
plt.title('1.2 plot Boundary for XA, YA')
learnerA.plotBoundary(XA, YA)'''

learnerB = logisticClassify2() # create "blank"learner
learnerB.classes = np.unique(YB) # define class labels using YA or YB
wts = np.array([0.5,-0.25,1]) # TODO: fill in values
learnerB.theta = wts # set the learner’s parameters
'''plt.figure()
plt.title('1.2 plot Boundary for XB, YB')
learnerB.plotBoundary(XB, YB)'''
#1.2 done

'''YhatA = learnerA.predict(XA)
YhatB = learnerB.predict(XB)
print('1.3 error for set A: ', learnerA.err(XA, YA))
print('1.3 error for set B: ', learnerB.err(XB, YB))
#1.3 done

plt.figure()
plt.title('1.4 verifing learnerA by plot on XA, YA')
ml.plotClassify2D(learnerA, XA, YA)
plt.figure()
plt.title('1.4 verifing learnerB by plot on XB, YB')
ml.plotClassify2D(learnerB, XB, YB)'''
#1.4 done
'''
learnerA.train(XA,YA,plotname = "Training learnerA")
print('Trained theta from learnerA: ', learnerA.theta)
learnerB.train(XB,YB,plotname = "Training learnerB")
print('Trained theta from learnerB: ', learnerB.theta)
'''

learnerA.train(XA,YA,plotname = "Training learnerA",regularization=True,alpha=0.06)
print('Trained theta from learnerA with regularization: ', learnerA.theta)
learnerB.train(XB,YB,plotname = "Training learnerB",regularization=True,alpha=0.06)
print('Trained theta from learnerB with regularization: ', learnerB.theta)

plt.show()
