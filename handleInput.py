import pandas as pd
import numpy as np
import FormatDataset as fd

def returnInputFileInfo(f):

    #get dataset, convert to csv
    line1 = f.readline().split(":")
    dataset = line1[-1].strip().replace("\n","")
    data = fd.convertToCSV(dataset)
    data = np.array(data)

    #relable
    data = fd.relableColumns(data)

    #get nominal feature indicies
    line2 = f.readline().split(":")
    nominal_features_indicies = line2[-1].strip()
    if(nominal_features_indicies != "None"):
        nominal_features_indicies= nominal_features_indicies.split(",")
        nominal_features_indicies[-1] = nominal_features_indicies[-1].replace("\n","")

    #map nominal feature indicies to new lables
    nominal_features_labels = [data[0,int(x)] for x in nominal_features_indicies]

    #get delete columns indicies
    line3 = f.readline().split(":")
    removeColumnIndicies = line3[-1].strip()
    if(removeColumnIndicies != "None"):
        removeColumnIndicies = removeColumnIndicies.split(",")
        removeColumnIndicies[-1] = removeColumnIndicies[-1].replace("\n","")

    #delete unwanted columns
    data = fd.removeColumns(data,removeColumnIndicies)

    #fill missing values
    line4 = f.readline().split(":")
    if(len(line4) > 1):
        missingValueString = line4[-1].strip()
        data = fd.fillMissingValues(data,missingValueString)

    #fill chosenAlgorithms boolean list
    chosenAlgorithms = [True,True,True,True,True]
    line5 = f.readline().split(":")
    if(line5[-1].strip() == "N" or line5[-1].strip() == "n"):
        chosenAlgorithms[0] = False
    line6 = f.readline().split(":")
    if(line6[-1].strip() == "N" or line6[-1].strip() == "n"):
        chosenAlgorithms[1] = False
    line7 = f.readline().split(":")
    if(line7[-1].strip() == "N" or line7[-1].strip() == "n"):
        chosenAlgorithms[2] = False
    line8 = f.readline().split(":")
    if(line8[-1].strip() == "N" or line8[-1].strip() == "n"):
        chosenAlgorithms[3] = False
    line9 = f.readline().split(":")
    if(line9[-1].strip() == "N" or line9[-1].strip() == "n"):
        chosenAlgorithms[4] = False

    return chosenAlgorithms,nominal_features_labels,data
