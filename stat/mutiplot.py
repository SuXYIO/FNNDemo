#!/usr/bin/env python3
'''
MutiPlot
Simple example of using two output CSV files to plot one graph.

Usage
    python3 mutiplot.py [filepath0] [filepath1] ...
    or
    ./mutiplot.py [filepath0] [filepath1] ...
        "filepath" is the path to the CSV formatted file. 
'''
import sys
import matplotlib.pyplot as plt
import pandas as pd

csvfilename = []
if len(sys.argv) <= 1:
    print('ERROR: too few arguments. ')
    exit(-1)
for i in range(len(sys.argv) - 1):
    csvfilename.append(sys.argv[i + 1])

df = []
ln = []
print(csvfilename)
for i in range(len(sys.argv) - 1):
    df.append(pd.read_csv(csvfilename[i]))
    ln.append(df[i].index)
# create plot
for i in range(len(sys.argv) - 1):
    plt.plot(ln[i], df[i]['f_w'], label = f'f_w{i}')
    plt.plot(ln[i], df[i]['f_b'], label = f'f_b{i}')
    plt.plot(ln[i], df[i]['g_w'], label = f'g_w{i}')
    plt.plot(ln[i], df[i]['g_b'], label = f'g_b{i}')
    plt.plot(ln[i], df[i]['l'], label = f'l{i}')

plt.title('MutiLinearRegressionPlot')
plt.xlabel('iter')
plt.ylabel('value')
plt.legend()

plt.show()
exit(0)
