## The DecisionTree class.

import collections
import math
import random
import sys

class DecisionTree:

    def __init__(self, trainingSet, size, depth):
        self.trainingSet = trainingSet
        self.size = size
        self.depth = depth
        self.feature = -1
        self.threshold = -1
        self.left = None
        self.right = None
        self.leaf = False
        self.result = -1
        self.buildDecTree()

    # Will create a decision tree.
    def buildDecTree(self):
        if self.depth == 1 or self.size <= 1:
            self.leaf = True
            self.result = self.popular()
        else:
            feats = self.featureSampling()
            infoGainF = -1*sys.maxsize
            for f in feats:
                sortedVectors = {}
                for vector in self.trainingSet:
                    sortedVectors.update({self.trainingSet[vector][0][f]:self.trainingSet[vector]})
                sortedVectors = collections.OrderedDict(sorted(sortedVectors.items()))
                T = []
                prev = sortedVectors.popitem(False)
                prev = prev[0]
                for val in sortedVectors:
                    temp = (prev + val)/2.0
                    T.append(temp)
                    prev = val
                bestThreshold = -1
                infoGainT = -1*sys.maxsize
                for t in T:
                    currInfoGainT = self.informationGain(f, t)
                    if currInfoGainT > infoGainT:
                        infoGainT = currInfoGainT
                        bestThreshold = t
                if infoGainT > infoGainF:
                    infoGainF = infoGainT
                    self.feature = f
                    self.threshold = bestThreshold
            setL = {}
            setR = {}
            setLsize = 0
            setRsize = 0
            for vector in self.trainingSet:
                if self.trainingSet[vector][0][self.feature] <= self.threshold:
                    setL.update({vector:self.trainingSet[vector]})
                    setLsize = setLsize + 1
                else:
                    setR.update({vector:self.trainingSet[vector]})
                    setRsize = setRsize + 1
            self.left = DecisionTree(setL, setLsize, self.depth-1)
            self.right = DecisionTree(setR, setRsize, self.depth-1)

    # Computes the information gain on the trainingSet using the given feature f and threshold t.
    def informationGain(self, f, t):
        setL = {}
        setR = {}
        setLsize = 0
        setRsize = 0
        for vector in self.trainingSet:
            if self.trainingSet[vector][0][f] <= t:
                setL.update({vector:self.trainingSet[vector]})
                setLsize = setLsize + 1
            else:
                setR.update({vector:self.trainingSet[vector]})
                setRsize = setRsize + 1
        return self.entropy(self.trainingSet) - ((setLsize/(self.size+0.0))*self.entropy(setL) +
                                                 (setRsize/(self.size+0.0))*self.entropy(setR))

    # Computes the entropy of the given set of data S.
    def entropy(self, S):
        tot = 0.0
        notSpam = 0.0
        spam = 0.0
        for vector in S:
            if S[vector][1][0] == '0':
                notSpam = notSpam + 1
            else:
                spam = spam + 1
            tot = tot + 1
        notSpam = notSpam/tot
        spam = spam/tot
        if (spam<0) or (notSpam<0):
            print("ERROR")
        if notSpam > 0.0:
            notSpam = notSpam*math.log(notSpam, 2)
        if spam > 0.0:
            spam = spam*math.log(spam, 2)
        return -1*(notSpam + spam)

    # Will return the class of the majority of the training set (0 if not spam, 1 if spam).
    def popular(self):
        notSpam = 0
        spam = 0
        for vector in self.trainingSet:
            if self.trainingSet[vector][1][0] == '0':
                notSpam = notSpam + 1
            else:
                spam = spam + 1
        if spam > notSpam:
            return 1
        else:
            return 0

    # Will return a list of [hyperparamter] random features.
    def featureSampling(self):
        i = 0
        samples = []
        while i<16:
            #r = random.randint(0,31)
            r = random.randint(0,35)
            #r = random.randint(0,104)
            if r not in samples:
                samples.append(r)
                i = i + 1
        return samples
