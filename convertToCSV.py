#converts whitespace seperated files to CSV files, and removes unwanted columns
#BUG: wont work on non csv files that have a comma in its first row

import sys
import csv
import shlex
import numpy as np

def convertToCSV(filename):
    data = []
    if(isCSV(filename)):
        with open(filename, 'r') as f:
            for row in f:
                data.append(row.split(','))
        return data
    with open(filename, 'r') as f:
        for row in f:
            data.append(shlex.split(row))
    return data

def isCSV(filename):
    with open(filename, 'r') as f:
        for row in f:
            if ',' in row:
                return True
            return False

if(len(sys.argv) < 2):
    print("Usage: python3 convertToCSV.py <filename> <delete columns indicies>")
    sys.exit()

data = np.array(convertToCSV(sys.argv[1]))

#which columns are kept?
keptColumns = [i for i in range(0,len(data.T))]
if(len(sys.argv) > 2):
    for i in sys.argv[2:]:
        keptColumns.remove(int(i))

#write only the wanted columns
data = data[:,keptColumns]

with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)
