import sys
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Variables
equityType = sys.argv[1]
stockName = sys.argv[2]
myStart = '2024-01-01'
myEnd = '2025-01-01'
sectorName = "QQQ"

# Folders
stocksFolder = './stocks/'
imagesFolder = './PNGS/'
txtFolder = './TXT/'

# Functions
def makeRepo(folder):
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
            print("created folder:", folder)
        except Exception as e:
            print("problem creating directory:", folder, "Error:", e)
    else:
        print("directory exists!", folder)

def saveToFile(content, filename):
    with open(os.path.join(txtFolder, filename), 'w') as file:
        file.write(content)

def downloadFile(tickerName, fileName, period):
    if not os.path.isfile(fileName):
        try:
            print("Downloading file:", fileName)
            downloadDF = yf.download(tickerName, start=myStart, end=myEnd, period=period)
            df = downloadDF.reset_index()
            # Calculate SMAs and other indicators
            for sma in [10, 30, 50, 100, 200, 300]:
                df[f'SMA{sma}'] = df['Close'].rolling(sma).mean()
                df[f'SMA{sma}PC'] = df[f'SMA{sma}'].pct_change()
            df['ClosePC'] = df['Close'].pct_change()
            df['VolumePC'] = df['Volume'].pct_change()
            df['VolLOG'] = np.log2(df['Volume'])
            df['CloseLOG'] = np.log2(df['Close'])
            df['VLOGPC'] = df['VolLOG'].pct_change()
            df['CLOGPC'] = df['CloseLOG'].pct_change()
            df.to_csv(fileName, index=False)
        except Exception as e:
            print("Problem downloading:", tickerName, "Error:", e)
    else:
        print("File exists!", fileName)

def calcFirstDerivative(myDataframe):
    a = myDataframe[1:].reset_index(drop=True)
    b = myDataframe[:-1].reset_index(drop=True)
    c = b['Close'] - a['Close']
    return c

def calcSecondDerivative(myDataframe):
    a = myDataframe[1:].reset_index(drop=True)
    b = myDataframe[:-1].reset_index(drop=True)
    c = b - a
    return c

def showUpDownDays():
    sns.set_style("white")
    totalDays = df3.count()
    myTitle = f"Distribution of UP/Down Days {stockName}: {totalDays} Days"
    myString = " "
    for i in range(0, int(df3.max()), 1):
        myDays = df3[(df3 == i) | (df3 == -i)].count()
        myString += f"\n{i}   [[  {myDays}  ]]   ,  {round(100 * (myDays / totalDays), 2)}% \n"
    df['difference'] = df['adj_close'].diff()
    df['pct_change'] = df['adj_close'].pct_change() * 100
    last_five_days = df.tail(5)[['adj_close', 'simple_rtn', 'difference', 'pct_change']]
    lineString = "Last five trading days:\n" + str(last_five_days)
    fig2, ax2 = plt.subplots()
    ax2.set_title(myTitle, fontsize=22)
    ax2.text(0.0, 1.0, myString, fontsize=14, verticalalignment='top', color='navy')
    ax2.text(0.0, 0.5, lineString, fontsize=14, verticalalignment='top', color='navy')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.legend(loc='lower right')
    plt.savefig(imagesFolder + f'text_graph_{stockName}_{myStart}_{myEnd}.png')

    # Save the textual information to a .txt file
    text_content = f"{myTitle}\n{myString}\n{lineString}"
    saveToFile(text_content, f'{stockName}_up_down_days.txt')

# Ensure directories exist
makeRepo(stocksFolder)
makeRepo(imagesFolder)
makeRepo(txtFolder)

# Download necessary data
tickers = [stockName, 'qqq', 'gold', 'spy', 'iwm', 'eem', 'tlt', 'dia', 'btc-usd', 'ung', 'uup', 'fxb', 'eur', 'uso', 'fxi', 'slv', 'kre', 'xlf', 'xlb', 'xle', 'xlp', 'xlu', 'xli', 'ibb', 'xlv', 'xly', 'xrt', 'xhb', 'xle', 'xop', 'xlk']
for ticker in tickers:
    downloadFile(ticker, os.path.join(stocksFolder, f"{ticker}_{myStart}_{myEnd}.csv"), '1d')

# Load the data
df = pd.read_csv(os.path.join(stocksFolder, f"{stockName}_{myStart}_{myEnd}.csv"))
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close': 'adj_close'}, inplace=True)
df['simple_rtn'] = df.adj_close.pct_change()
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()
df2 = df['simple_rtn'].copy(deep=True)
df2[df2 > 0] = 1
df2[df2 < 0] = -1
df3 = df2.copy(deep=True)
tally = 1
old = 1
runningSum = 0
for value in df3:
    if math.isnan(value):
        print("skipping a nan")
    else:
        combined = int(old) + int(value)
        if combined == 0:
            runningSum = 0
        elif combined < 0:
            runningSum = runningSum - 1
        elif combined > 0:
            runningSum = runningSum + 1
        tally = tally + 1
        old = value
        df3.iloc[tally - 1] = runningSum

# Generate the textual graph and save it
showUpDownDays()

print("exiting OK")
sys.exit()

