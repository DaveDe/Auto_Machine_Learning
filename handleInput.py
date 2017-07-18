import pandas as pd
import numpy as np
import FormatDataset as fd

def returnInputFileInfo(f):

    testDataFile = False
    testData = []

    #get datasets, convert to csv
    line1 = f.readline().split(":")
    trainDataset = line1[-1].strip().replace("\n","")
    trainData = fd.convertToCSV(trainDataset)
    trainData = np.array(trainData)
    trainData = fd.relableColumns(trainData)

    line2 = f.readline().split(":")
    if(line2[-1].strip().replace("\n","") != ""):
        testDataFile = True
        testDataset = line2[-1].strip().replace("\n","")
        testData = fd.convertToCSV(testDataset)
        testData = np.array(testData)
        testData = fd.relableColumns(testData)

    outputPredictions = True
    line3 = f.readline().split(":")
    if(line3[-1].strip() == "N" or line3[-1].strip() == "n"):
        outputPredictions = False

    #get nominal feature indicies
    line4 = f.readline().split(":")
    nominal_features_indicies = line4[-1].strip()
    if(nominal_features_indicies != ""):
        nominal_features_indicies= nominal_features_indicies.split(",")
        nominal_features_indicies[-1] = nominal_features_indicies[-1].replace("\n","")

    #map nominal feature indicies to new lables
    nominal_features_labels = [trainData[0,int(x)] for x in nominal_features_indicies]

    #get delete columns indicies
    line5 = f.readline().split(":")
    removeColumnIndicies = line5[-1].strip()
    if(removeColumnIndicies != ""):
        removeColumnIndicies = removeColumnIndicies.split(",")
        removeColumnIndicies[-1] = removeColumnIndicies[-1].replace("\n","")

    #delete unwanted columns
    trainData = fd.removeColumns(trainData,removeColumnIndicies)
    if(testDataFile):
        testData = fd.removeColumns(testData, removeColumnIndicies)

    #fill missing values
    line6 = f.readline().split(":")
    if(len(line6) > 1):
        missingValueString = line6[-1].strip()
        trainData = fd.fillMissingValues(trainData,missingValueString)
        if(testDataFile):
            testData = fd.fillMissingValues(testData,missingValueString)

    #identify and drop columns with high covariate shift?
    perfCovariateShift = True
    line7 = f.readline().split(":")
    if(line7[-1].strip() == "N" or line7[-1].strip() == "n"):
        perfCovariateShift = False

    #fill chosenAlgorithms boolean list
    chosenAlgorithms = [True,True,True,True,True,True]
    line8 = f.readline().split(":")
    if(line8[-1].strip() == "N" or line8[-1].strip() == "n"):
        chosenAlgorithms[0] = False
    line9 = f.readline().split(":")
    if(line9[-1].strip() == "N" or line9[-1].strip() == "n"):
        chosenAlgorithms[1] = False
    line10 = f.readline().split(":")
    if(line10[-1].strip() == "N" or line10[-1].strip() == "n"):
        chosenAlgorithms[2] = False
    line11 = f.readline().split(":")
    if(line11[-1].strip() == "N" or line11[-1].strip() == "n"):
        chosenAlgorithms[3] = False
    line12 = f.readline().split(":")
    if(line12[-1].strip() == "N" or line12[-1].strip() == "n"):
        chosenAlgorithms[4] = False
    line13 = f.readline().split(":")
    if(line13[-1].strip() == "N" or line13[-1].strip() == "n"):
        chosenAlgorithms[5] = False

    return chosenAlgorithms,perfCovariateShift,nominal_features_labels,trainData,testData,outputPredictions
