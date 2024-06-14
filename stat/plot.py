#!/usr/bin/env python3
'''
LinearRegressionPlot
Simple example of using the output CSV file of LR to plot.

Usage
    python3 plot.py [filepath]
    or
    ./plot.py [filepath]
        "filepath" is the path to the CSV formatted file LR generated. 
        If "filepath" empty, the program will search in the current dir for "*.csv". 
        It uses the only one if there is only one "*.csv" file. 
        It prompts the user which one to use if there is more than one or no "*.csv" file. 
'''
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) > 1:
    csvfilename = sys.argv[1]
else:
    PATTERN = ".csv"
    csvfiles = []
    for f in os.listdir("."):
        if f.endswith(PATTERN):
            csvfiles.append(f)
    if len(csvfiles) > 1:
        # muti csv, ask for one
        print(f'Found mutiple CSV files: \n{print(csvfiles)}\n')
        csvfilename = input('Choose one CSV file\n> ')
    elif len(csvfiles) == 1:
        csvfilename = csvfiles[0]
    else:
        # no csv
        csvfilename = input('Enter filepath\n> ')

df = pd.read_csv(csvfilename)
ln = df.index

plt.plot(ln, df['f_w'], label = 'f_w')
plt.plot(ln, df['f_b'], label = 'f_b')
plt.plot(ln, df['g_w'], label = 'g_w')
plt.plot(ln, df['g_b'], label = 'g_b')
plt.plot(ln, df['l'], label = 'l')
plt.plot(ln, df['grad_w'], label = 'grad_w')
plt.plot(ln, df['grad_b'], label = 'grad_b')

plt.title('LinearRegressionPlot')
plt.xlabel('iter')
plt.ylabel('value')
plt.legend()

plt.show()
