import sys
import csv
import shlex
import numpy as np
import statistics as stat
import pandas as pd
from sklearn.model_selection import train_test_split

def convertToCSV(filename):
    data = []
    if(isCSV(filename)):
        with open(filename, 'r') as f:
            for row in f:
                row = row.split(',')
                if(row[-1][-1] == "\n"):
                    row[-1] = row[-1][:-1]
                data.append(row)
        return data
    with open(filename, 'r') as f:
        for row in f:
            row = shlex.split(row)
            if(row[-1][-1] == "\n"):
                row[-1] = row[-1][:-1]
            data.append(row)
    return data

def isCSV(filename):
    with open(filename, 'r') as f:
        for row in f:
            if ',' in row:
                return True
            return False

#relable columns (X0,X1,...XN)
def relableColumns(data):
    colIndex = 0
    for position,label in enumerate(data[0,:]):
        if(label != "y" and label != "Y"):
            data[0,position] = "X"+str(colIndex)
            colIndex += 1
        else:
            data[0,position] = "Y" #ensure Y is capital
    return data

def removeColumns(data,removeColumnsIndicies):
    keptColumns = [i for i in range(0,len(data.T))]
    for i in removeColumnsIndicies:
        keptColumns.remove(int(i))
    data = data[:,keptColumns]
    return data

#fill missing values with median
def fillMissingValues(data,missingValueString):
    #first find median of all columns, excluding the missing character entries
    medians = []
    for col in data.T[:,1:]:
        newCol = [x for x in col if x != missingValueString]
        medians.append(stat.median(newCol))
    #next fill missing characters with medians
    for i,col in enumerate(data.T):
        for j,item in enumerate(col):
            if(item == missingValueString):
                data[j,i] = medians[i]
    return data

#add columns to a and b so they have the same labled columns and same shape
def addBlankColumns(a,b):

    ACols = a.columns.tolist()
    BCols = b.columns.tolist()

    #find columns that need to be added to a and b
    addToA = [x for x in BCols if x not in ACols]
    addToB = [x for x in ACols if x not in BCols]

    #add needed columns to a and b, with data filled with 0's
    for col in addToA:
        a[col] = 0
    for col in addToB:
        b[col] = 0

    #ensure a and b have same column names in same order
    b.columns = a.columns

    return a,b

def noTestDataFile(X,Y,nominal_features_labels):

    #convert nominal to dummy
    if(len(nominal_features_labels) > 0):
        for i in nominal_features_labels:
            dummy_matrix = pd.get_dummies(X[i])
            X = pd.concat([X,dummy_matrix], axis=1)
        X.drop(nominal_features_labels, axis=1, inplace = True)

    #relabel column names to overwrite duplicate names
    newCols = []
    for i,j in enumerate(X.columns):
        newCols.append("X"+str(i))
    X.columns = newCols

    return train_test_split(X, Y, test_size=0.33,random_state=10)

def convertNominalToDummy(trainingData,testData,nominal_features_labels):

    all_indicies = [x for x in range(0,len(trainingData[0,:]))]
    featureIndicies = [i for i,j in enumerate(trainingData[0,:]) if j != "Y"]
    Y_index = [x for x in all_indicies if x not in featureIndicies]

    X_train = trainingData[:,featureIndicies]
    Y_train = trainingData[:,Y_index[0]]

    X_train = pd.DataFrame(data=X_train[1:,:], columns=X_train[0,:])
    Y_train = pd.DataFrame(Y_train[1:])

    if(len(testData) == 0):
        return noTestDataFile(X_train,Y_train,nominal_features_labels)

    #extract test data X and Y
    all_indicies = [x for x in range(0,len(testData[0,:]))]
    featureIndicies = [i for i,j in enumerate(testData[0,:]) if j != "Y"]
    Y_index = [x for x in all_indicies if x not in featureIndicies]

    X_test = testData[:,featureIndicies]

    if(len(Y_index) == 0):
        Y_test = [0 for x in range(0,len(X_test[:,0]))] #placeholder Y if test file doesnt have Y
        print("Test file doesnt have Y. Expect R^2 to be 0. Make sure to output predictions")
    else:
        Y_test = testData[:,Y_index[0]]

    X_test = pd.DataFrame(data=X_test[1:,:], columns=X_test[0,:])
    Y_test = pd.DataFrame(Y_test[1:])

    #convert nominal features into dummy matricies
    #make sure when training set or test set gets an extra column, a blank one is addded for the other
    if(len(nominal_features_labels) > 0):
        for i in nominal_features_labels:
            dummy_matrix_train = pd.get_dummies(X_train[i])
            dummy_matrix_test = pd.get_dummies(X_test[i])
            dummy_matrix_train, dummy_matrix_test = addBlankColumns(dummy_matrix_train,dummy_matrix_test)
            X_train = pd.concat([X_train,dummy_matrix_train], axis=1)
            X_test = pd.concat([X_test,dummy_matrix_test], axis=1)

        X_train.drop(nominal_features_labels, axis=1, inplace = True)
        X_test.drop(nominal_features_labels, axis=1, inplace = True)

    #relable X_train and X_test so there are no duplicate column names
    newCols = []
    for i,j in enumerate(X_train.columns):
        newCols.append("X"+str(i))
    X_train.columns = newCols
    X_test.columns = newCols

    return X_train,X_test,Y_train,Y_test

def writeCSV(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow([row])
