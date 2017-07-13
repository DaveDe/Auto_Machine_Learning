import pandas as pd
import numpy as np
import FormatDataset as fd

def returnInputFileInfo(f):

    testDataFile = False

    #get datasets, convert to csv
    line1 = f.readline().split(":")
    trainDataset = line1[-1].strip().replace("\n","")
    trainData = fd.convertToCSV(trainDataset)
    trainData = np.array(trainData)
    trainData = fd.relableColumns(trainData)

    line2 = f.readline().split(":")
    if(len(line2) > 1):
        testDataFile = True
        testDataset = line2[-1].strip().replace("\n","")
        testData = fd.convertToCSV(testDataset)
        testData = np.array(testData)
        testData = fd.relableColumns(testData)

    #get nominal feature indicies
    line3 = f.readline().split(":")
    nominal_features_indicies = line3[-1].strip()
    if(nominal_features_indicies != "None"):
        nominal_features_indicies= nominal_features_indicies.split(",")
        nominal_features_indicies[-1] = nominal_features_indicies[-1].replace("\n","")

    #map nominal feature indicies to new lables
    nominal_features_labels = [data[0,int(x)] for x in nominal_features_indicies]

    #get delete columns indicies
    line4 = f.readline().split(":")
    removeColumnIndicies = line4[-1].strip()
    if(removeColumnIndicies != "None"):
        removeColumnIndicies = removeColumnIndicies.split(",")
        removeColumnIndicies[-1] = removeColumnIndicies[-1].replace("\n","")

    #delete unwanted columns
    trainData = fd.removeColumns(trainData,removeColumnIndicies)
    if(testDataFile):   
        testData = fd.removeColumns(testData, removeColumnIndicies)

    #fill missing values
    line5 = f.readline().split(":")
    if(len(line5) > 1):
        missingValueString = line5[-1].strip()
        trainData = fd.fillMissingValues(trainData,missingValueString)
        if(testDataFile):
            testData = fd.fillMissingValues(testData,missingValueString)


    #fill chosenAlgorithms boolean list
    chosenAlgorithms = [True,True,True,True,True]
    line6 = f.readline().split(":")
    if(line6[-1].strip() == "N" or line6[-1].strip() == "n"):
        chosenAlgorithms[0] = False
    line7 = f.readline().split(":")
    if(line7[-1].strip() == "N" or line7[-1].strip() == "n"):
        chosenAlgorithms[1] = False
    line8 = f.readline().split(":")
    if(line8[-1].strip() == "N" or line8[-1].strip() == "n"):
        chosenAlgorithms[2] = False
    line9 = f.readline().split(":")
    if(line9[-1].strip() == "N" or line9[-1].strip() == "n"):
        chosenAlgorithms[3] = False
    line10 = f.readline().split(":")
    if(line10[-1].strip() == "N" or line10[-1].strip() == "n"):
        chosenAlgorithms[4] = False

    return chosenAlgorithms,nominal_features_labels,trainData, testData
