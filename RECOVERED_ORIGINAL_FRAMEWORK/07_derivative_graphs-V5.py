import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.api as sm

# Variables
equityType = 'Stock'
#stockName = 'AMZN'
stockName = sys.argv[1]
# equityType = sys.argv[2]

myStart = '2024-01-01'
myEnd = '2025-01-01'
sectorName = "QQQ"

stocksFolder = './stocks/'
imagesFolder = './PNGS/'
pdfsFolder = './PDFS/'
period = '1d'
stockFile = stocksFolder + stockName + '_' + myStart + '_' + myEnd + ".csv"

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

def downloadFile(tickerName, fileName, period):
    if not os.path.isfile(fileName):
        try:
            print("download the file:", fileName)
            downloadDF = yf.download(tickerName, start=myStart, end=myEnd, period=period)
            df = downloadDF.reset_index()
            # Calculating SMAs and percentage changes
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
        print("file exists!", fileName)

def calcFirstDerivative(myDataframe):
    a = myDataframe[1:].reset_index(drop=True)
    b = myDataframe[:-1].reset_index(drop=True)
    c = b - a
    return c

def calcSecondDerivative(myDataframe):
    a = myDataframe[1:].reset_index(drop=True)
    b = myDataframe[:-1].reset_index(drop=True)
    c = b - a
    return c

# Ensure directories exist
makeRepo(stocksFolder)
makeRepo(imagesFolder)
downloadFile(stockName, stockFile, period)

# Load Data
df = pd.read_csv(stockFile)

# Function to draw deviation lines
def drawLines(ax, x, y, deviations, color):
    for i in deviations:
        ax.plot(x, y + (float(i) * np.std(y)), color=color, linewidth=0.5, linestyle='-')
        ax.plot(x, y - (float(i) * np.std(y)), color=color, linewidth=0.5, linestyle='-')

# Plot first derivative
first_derivative = calcFirstDerivative(df['Close'])
df['FirstDerivative'] = first_derivative

plt.figure(figsize=(10, 6))
sns.regplot(x=df.index, y=df['FirstDerivative'], data=df, ci=None, marker='.', color='gold', scatter_kws={'s': 75})
sns.lineplot(x=df.index, y=df['FirstDerivative'], data=df, color='navy', linewidth=1)
drawLines(plt.gca(), df.index, df['FirstDerivative'], ['.25', '.5', '.75', '1.25', '1.5', '1.75', '2.25', '2.5', '2.75', '3.25', '3.5', '3.75'], 'lightgrey')
drawLines(plt.gca(), df.index, df['FirstDerivative'], ['1', '2', '3', '4'], 'black')
plt.title(f"First Derivative {stockName.upper()} {myStart} to {myEnd}")
plt.savefig(os.path.join(imagesFolder, f'first_derivative_{stockName}_{myStart}_{myEnd}.png'))
plt.show()

# Plot second derivative
second_derivative = calcSecondDerivative(first_derivative)
df['SecondDerivative'] = second_derivative

plt.figure(figsize=(10, 6))
sns.regplot(x=df.index, y=df['SecondDerivative'], data=df, ci=None, marker='.', color='red', scatter_kws={'s': 75})
sns.lineplot(x=df.index, y=df['SecondDerivative'], data=df, color='green', linewidth=1)
drawLines(plt.gca(), df.index, df['SecondDerivative'], ['.25', '.5', '.75', '1.25', '1.5', '1.75', '2.25', '2.5', '2.75', '3.25', '3.5', '3.75'], 'lightgrey')
drawLines(plt.gca(), df.index, df['SecondDerivative'], ['1', '2', '3', '4'], 'black')
plt.title(f"Second Derivative {stockName.upper()} {myStart} to {myEnd}")
plt.savefig(os.path.join(imagesFolder, f'second_derivative_{stockName}_{myStart}_{myEnd}.png'))
plt.show()

# Plot mini dataframe statistics
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
CloseString = df['Close'].tail(6).head(5).to_string(index=False)
VolumeString = df['Volume'].tail(6).head(5).to_string(index=False)
DateString = df['Date'].tail(6).head(5).to_string(index=False)
OpenString = df['Open'].tail(6).head(5).to_string(index=False)
DailyVolatilityString = (df['High'] - df['Low']).tail(6).head(5).to_string(index=False)
ax.text(0, 1, f'Last five Periods mini dataframe: {stockName}', size=12, color='black')
ax.text(0, 0, f'last 5 closingPrice:\n{CloseString}', size=12, color='navy')
ax.text(0.5, 0.5, f'Trading Volume:\n{VolumeString}', size=12, color='green')
ax.text(0.55, 0, f'OpeningPrice:\n{OpenString}', size=12, color='black')
ax.text(0.22, 0.3, f'Dates:\n{DateString}', size=12, color='Purple')
ax.text(0, 0.5, f'DailyVolatility price range:\n{DailyVolatilityString}', size=12, color='red')
plt.savefig(os.path.join(imagesFolder, f'mini_dataframe_stats_{stockName}_{myStart}_{myEnd}.png'))
plt.show()

# Plot SMA statistics
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
SMA_strings = {sma: df[f'SMA{sma}'].tail(6).head(5).to_string(index=False) for sma in [10, 30, 50, 100, 200]}
ax.text(0, 1, f'SMAs last 5 Periods aka current critical target prices: {stockName}', size=12, color='black')
for i, (sma, color) in enumerate(zip([10, 30, 50, 100, 200], ['navy', 'green', 'black', 'Purple', 'red'])):
    ax.text(i % 2 * 0.5, (1 - (i // 2) * 0.3), f'SMA{sma}:\n{SMA_strings[sma]}', size=12, color=color)
plt.savefig(os.path.join(imagesFolder, f'mini_dataframe_SMAs_{stockName}_{myStart}_{myEnd}.png'))
plt.show()

# Plot distribution and Q-Q plot
df['log_rtn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()
mu, sigma = df['log_rtn'].mean(), df['log_rtn'].std()
r_range = np.linspace(df['log_rtn'].min(), df['log_rtn'].max(), num=1000)
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
sns.histplot(df['log_rtn'], kde=False, stat='density', ax=ax[0])
ax[0].plot(r_range, norm_pdf, 'g', lw=2, label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].set_title(f'Distribution of {stockName} returns', fontsize=16)
ax[0].legend(loc='upper left')

sm.qqplot(df['log_rtn'].values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize=16)
plt.savefig(os.path.join(imagesFolder, f'distribution_and_qqplot_{stockName}_{myStart}_{myEnd}.png'))
plt.show()

