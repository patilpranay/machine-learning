# A T Random Forests implementation.

import collections
import csv
import math
import numpy as np
import random
import scipy.io as sio
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import Imputer

from DecisionTree import DecisionTree

def splitData(data, labels, splitPercentage):
    """ Splits the data (and labels) into training and validation sets. """
    # get number of samples and features
    samples = data.shape[0]
    features = data.shape[1]

    # get number of samples that will be in the training and validation sets
    trainingSize = int(splitPercentage*samples)
    validationSize = samples - trainingSize

    # generate empty matrices to hold training dataset and labels
    X_train = np.zeros((trainingSize, features))
    labels_train = np.zeros((trainingSize, 1))

    # generate empty matrices to hold validation dataset and labels
    X_val = np.zeros((validationSize, features))
    labels_val = np.zeros((validationSize, 1))

    # indices that we have already seen for training set
    seen = []

    # create the training set
    for i in range(trainingSize):
        temp_index = random.randint(0, samples)
        while ((temp_index in seen) or (temp_index == samples)):
            temp_index = random.randint(0, samples)
        X_train[i] = data[temp_index]
        labels_train[i] = labels[temp_index]
        seen.append(temp_index)

    # create the validation set - we don't need to shuffle validation set
    count = 0
    for i in range(samples):
        if (i not in seen):
            X_val[count] = data[i]
            labels_val[count] = labels[i]
            count = count + 1

    return X_train, labels_train, X_val, labels_val

# Predicts the class of an email given a decision tree.
def predict(vector, tree):
    if tree.leaf:
        return tree.result
    if vector[tree.feature] <= tree.threshold:
        return predict(vector, tree.left)
    else:
        return predict(vector, tree.right)

# Implements bagging by selecting bagSize emails from S and returning them.
def bagging(S, setSize, bagSize):
    baggedSet = {}
    for i in range(int(bagSize)):
        c = random.randint(1, setSize)
        baggedSet.update({c:S[c]})
    return baggedSet

# Loads and processes the census data.
def loadAndProcessCensusData():
    data = np.zeros((32724, 8))
    noncat_data = np.zeros((32724, 6))
    labels = np.zeros((32724, 1))
    header = None
    categorical = ['workclass', 'education', 'marital-status', 'occupation', \
                   'relationship', 'race', 'sex', 'native-country']
    seen = {'workclass':[], 'education':[], 'marital-status':[], \
            'occupation':[], 'relationship':[], 'race':[], 'sex':[], \
            'native-country':[]}
    with open('train_data.csv') as csvfile:
        labelreader = csv.reader(csvfile, dialect='excel')
        count = -1
        for row in labelreader:
            if (count == -1):
                header = row
            else:
                c1 = 0
                c2 = 0
                for i in range(14):
                    if (header[i] in categorical):
                        if (row[i] == '?'):
                            data[count][c1] = '-1'
                        else:
                            if (row[i] not in seen[header[i]]):
                                seen[header[i]].append(row[i])
                            data[count][c1] = seen[header[i]].index(row[i])
                        c1 = c1 + 1
                    else:
                        if (row[i] == '?'):
                            noncat_data[count][c2] = -1
                        else:
                            noncat_data[count][c2] = int(row[i])
                        c2 = c2 + 1
                labels[count][0] = int(row[14])
            count = count + 1

    fix = Imputer(-1, "most_frequent", axis=0)
    data = fix.fit_transform(data)
    noncat_data = fix.fit_transform(noncat_data)

    data_test = np.zeros((16118, 8))
    noncat_data_test = np.zeros((16118, 6))
    seen_test = {'workclass':[], 'education':[], 'marital-status':[], \
                 'occupation':[], 'relationship':[], 'race':[], 'sex':[], \
                 'native-country':[]}
    with open('test_data.csv') as csvfile:
        labelreader = csv.reader(csvfile, dialect='excel')
        count = -1
        for row in labelreader:
            if (count == -1):
                header = row
            else:
                c1 = 0
                c2 = 0
                for i in range(14):
                    if (header[i] in categorical):
                        if (row[i] == '?'):
                            data_test[count][c1] = '-1'
                        else:
                            if (row[i] not in seen_test[header[i]]):
                                seen_test[header[i]].append(row[i])
                            data_test[count][c1] = seen_test[header[i]].index(row[i])
                        c1 = c1 + 1
                    else:
                        if (row[i] == '?'):
                            noncat_data_test[count][c2] = -1
                        else:
                            noncat_data_test[count][c2] = int(row[i])
                        c2 = c2 + 1
            count = count + 1

    data_test = fix.fit_transform(data_test)
    noncat_data_test = fix.fit_transform(noncat_data_test)

    data_list = []
    for i in range(data.shape[0]):
        entry = {}
        c1 = 0
        for j in range(14):
            if (header[j] in categorical):
                entry[header[j]] = seen[header[j]][int(data[i][c1])]
                c1 = c1 + 1
        data_list.append(collections.OrderedDict(sorted(entry.items())))

    data_list_test = []
    for i in range(data_test.shape[0]):
        entry = {}
        c1 = 0
        for j in range(14):
            if (header[j] in categorical):
                entry[header[j]] = seen_test[header[j]][int(data_test[i][c1])]
                c1 = c1 + 1
        data_list_test.append(collections.OrderedDict(sorted(entry.items())))


    v = DictVectorizer(sparse=False)
    X = v.fit_transform(data_list)
    X_test = v.transform(data_list_test)

    result = np.concatenate((X, noncat_data), axis=1)
    result_test = np.concatenate((X_test, noncat_data_test), axis=1)

    return result, labels, result_test

# Load in the data for spam.
#data = sio.loadmat('spam_data.mat')
data = sio.loadmat('spam_data_pranay.mat')
train_labels = data['training_labels'].T
train_data = data['training_data']
test_data = data['test_data']

# Load in the data for census.
#train_data, train_labels, test_data = loadAndProcessCensusData()

X_train, lab_train, X_val, lab_val = splitData(train_data, train_labels, 0.8)
#X_train = train_data
#lab_train = train_labels

trainingSet = {}
for i in range(X_train.shape[0]):
    trainingSet.update({i+1:(X_train[i], [str(int(lab_train[i]))])})

temp = {}
#for i in range(lab_val.shape[0]):
#    temp.update({i+1:int(lab_val[i])})

for i in range(lab_train.shape[0]):
    temp.update({i+1:int(lab_train[i])})

#valVectors = test_data
#valVectors = X_val
valVectors = X_train

Tvals = [1]
for T in Tvals:
    trees = {}
    for i in range(T):
        b = bagging(trainingSet, len(trainingSet), \
                    math.floor(0.9*(len(trainingSet))))
        tree = DecisionTree(b, len(b), 40)
        trees.update({i+1:tree})

    count = 1
    total = 0
    errors = 0
    results = []
    for vector in valVectors:
        notSpam = 0
        spam = 0
        vClass = -1
        for tree in trees:
            p = predict(vector, trees[tree])
            if p==0:
                notSpam = notSpam + 1
            else:
                spam = spam + 1
        if notSpam > spam:
            vClass = 0
        elif spam > notSpam:
            vClass = 1
        else:
            vClass = 0
        if vClass != temp[count]:
        #if vClass != 1:
            errors = errors + 1
        results.append(vClass)
        count = count + 1
        total = total + 1

    with open('results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', \
                 quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Category'])
        for i in range(len(results)):
            first = i+1
            second = int(results[i])
            writer.writerow([first, second])

    rate = (errors/(total+0.0))*100
    print("For T=" + str(T) + ", the accuracy is " + str(100 - rate) + "%.")
