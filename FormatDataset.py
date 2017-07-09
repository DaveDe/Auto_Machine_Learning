#converts whitespace seperated files to CSV files, removes unwanted columns,
#relabels labels, fills missing values
#BUG: wont work on non csv files that have a comma in its first row

import sys
import csv
import shlex
import numpy as np
import statistics as stat

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

#write to output.csv
def writeReformattedCSV(data):
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
