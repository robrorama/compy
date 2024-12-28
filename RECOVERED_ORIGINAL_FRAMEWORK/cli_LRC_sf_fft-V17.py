# -*- coding: utf-8 -*-
'''
Original file is located at
    https://colab.research.google.com/drive/1lEJAIBU2qgVMv0eL_SqNKJjc0_e2lFhO

The main thrust is currently to show Linear Regression Channel
Both for current stock, its percentage change, the same relative to its sector.
Later will show sharkfins
currently commented out 
do both relative?
Need to add another function that will do all of the sectors and the rest 
Need to add back in the relative comparison 
1. do the second derivative chart
2. do the pdf maker for the full integrated image ?
3. the distance between the SMA's and the order, abdc 
4. make sure that it runs as independent script to generate all the datas.
'''

import sys
import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns

stockName = sys.argv[1] #@param {type:'string'}
myStart = '2019-01-01' #@param {type:'string'}
myEnd = '2025-01-01' #@param {type:'string'}
sectorName = "QQQ" #@param {type:'string'}
equityType = 'Stock' #@param ['ETF','Stock']

print("startDate:" + myStart)
print("endDate:" + myEnd)

# Variables
stocksFolder = './stocks/'
imagesFolder = './PNGS/'
period = '1d'

stockFile = stocksFolder + stockName + '_' + myStart + '_' + myEnd + ".csv"
sectorFile = stocksFolder + sectorName + '_' + myStart + '_' + myEnd + ".csv"
qqqFile = stocksFolder + 'qqq_' + myStart + '_' + myEnd + ".csv"
goldFile = stocksFolder + 'gold_' + myStart + '_' + myEnd + ".csv"
spyFile = stocksFolder + 'spy_' + myStart + '_' + myEnd + ".csv"
iwmFile = stocksFolder + 'iwm_' + myStart + '_' + myEnd + ".csv"
eemFile = stocksFolder + 'eem_' + myStart + '_' + myEnd + ".csv"
tltFile = stocksFolder + 'tlt_' + myStart + '_' + myEnd + ".csv"
diaFile = stocksFolder + 'dia_' + myStart + '_' + myEnd + ".csv"

def makeRepo(folder):
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
            print("created folder")
        except:
            print("problem creating directory:", folder)
    else:
        print("directory exists!", folder)

def downloadFile(tickerName, fileName, period):
    if not os.path.isfile(fileName):
        try:
            print("download the file:", fileName)
            downloadDF = yf.download(tickerName, start=myStart, end=myEnd, period=period)
            df = downloadDF
            df = df.reset_index()
            df['SMA30'] = df['Close'].rolling(30).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA100'] = df['Close'].rolling(100).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['SMA300'] = df['Close'].rolling(300).mean()
            df['SMA30PC'] = df['SMA30'].pct_change()
            df['SMA50PC'] = df['SMA50'].pct_change()
            df['SMA100PC'] = df['SMA100'].pct_change()
            df['SMA200PC'] = df['SMA200'].pct_change()
            df['ClosePC'] = df['Close'].pct_change()
            df['VolumePC'] = df['Volume'].pct_change()
            df['VolLOG'] = np.log2(df['Volume'])
            df['CloseLOG'] = np.log2(df['Close'])
            df['VLOGPC'] = df['VolLOG'].pct_change()
            df['CLOGPC'] = df['CloseLOG'].pct_change()
            df.to_csv(fileName)
        except:
            print("Problem downloading:", tickerName)
    else:
        print("file exists!", fileName)

# Trigger it:
makeRepo(stocksFolder)
makeRepo(imagesFolder)
downloadFile(stockName, stockFile, period)
downloadFile(sectorName, sectorFile, period)
downloadFile(sectorName, qqqFile, period)
downloadFile(sectorName, goldFile, period)
downloadFile(sectorName, spyFile, period)
downloadFile(sectorName, iwmFile, period)
downloadFile(sectorName, eemFile, period)
downloadFile(sectorName, tltFile, period)
downloadFile(sectorName, diaFile, period)

##### This is the main LRC graph #######
# Function to draw lines
def drawLines(deviations, color, x_rp, y_rp):
    for i in deviations:
        sns.lineplot(x=x_rp, y=y_rp + (float(i) * np.std(y_rp)), color=color, linewidth=.5, linestyle='-')
        sns.lineplot(x=x_rp, y=y_rp - (float(i) * np.std(y_rp)), color=color, linewidth=.5, linestyle='-')

# Main LRC graph
def plotLRC():
    df = pd.read_csv(stockFile)
    plt.rc("figure", figsize=(10, 8), dpi=300)
    plt.rc("font", size=14)

    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='lightblue', scatter_kws={'s':75})
    zp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='navy', scatter_kws={'s':5})

    y_rp = rp.get_lines()[0].get_ydata()
    x_rp = rp.get_lines()[0].get_xdata()

    deviations = ['.25', '.5', '.75', '1', '1.25', '1.5', '1.75']
    drawLines(deviations, 'lightgrey', x_rp, y_rp)
    deviations = ['1', '2', '3', '4', '5']
    drawLines(deviations, 'black', x_rp, y_rp)

    smas = ['SMA30', 'SMA50', 'SMA100', 'SMA200', 'SMA300']
    smaColors = ['g', 'orange', 'blue', 'purple', 'red']
    for i in range(len(smas)):
        sns.lineplot(x=df.index, y=smas[i], data=df, color=smaColors[i], linewidth=1.6)

    plt.title("LRC:  " + stockName.upper() + str(myStart) + "   to:" + str(myEnd))
    plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
    plt.savefig(imagesFolder + stockName + '_' + myStart + '_' + myEnd + '.png')
    plt.show()

plotLRC()

##### END main LRC graph #######

# Pie chart function
def makePieChart():
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{stockName}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{stockName}_recommendations.csv"

    try:
        recommendations = pd.read_csv(data_file)
        print(f"Recommendations for {stockName} found on disk.")
    except FileNotFoundError:
        stock = yf.Ticker(stockName)
        recommendations = stock.recommendations
        recommendations.to_csv(data_file)
        print(f"Recommendations for {stockName} downloaded and saved to disk.")

    buy_count = recommendations['strongBuy'].sum() + recommendations['buy'].sum()
    hold_count = recommendations['hold'].sum()
    sell_count = recommendations['sell'].sum() + recommendations['strongSell'].sum()

    labels = ["Buy", "Hold", "Sell"]
    sizes = [buy_count, hold_count, sell_count]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"Analyst Recommendations for {stockName}")
    plt.axis("equal")

    image_dir = f"images/{stockName}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{stockName}_analyst_recommendations.png')
    plt.savefig(image_file, dpi=300)

    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{stockName}_{today_str}_analyst_recommendations.png')
    plt.savefig(pngs_file, dpi=300)

    plt.show()

if equityType == 'Stock':
    print("equity Type is a stock ")
    print("making PieChart")
    makePieChart()
else:
    print(" equityType is ETF : Doing nothing")

print("exiting")
sys.exit()

