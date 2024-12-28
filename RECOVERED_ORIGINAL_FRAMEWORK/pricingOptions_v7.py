import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Input parameters
equityType = sys.argv[1]
stockName = sys.argv[2] 
numberWeeks = sys.argv[3] 
myStart = '2021-01-01'
myEnd = '2023-01-01'
sectorName = "QQQ" 

# Ensure necessary folders exist
stocksFolder = './stocks/'
imagesFolder = './PNGS/'
optionsBaseFolder = './options/'
targetPricesBaseFolder = './TARGET_PRICES_OPTIONS/'
period = '1d'

def makeRepo(folder):
    if not os.path.isdir(folder):
        try:
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        except:
            print(f"Problem creating directory: {folder}")
    else:
        print(f"Directory exists: {folder}")

def downloadFile(tickerName, fileName, period):
    if not os.path.isfile(fileName):
        try:
            print(f"Downloading the file: {fileName}")
            downloadDF = yf.download(tickerName, start=myStart, end=myEnd, period=period)
            df = downloadDF.reset_index()
            df['SMA10'] = df['Close'].rolling(10).mean()
            df['SMA30'] = df['Close'].rolling(30).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA100'] = df['Close'].rolling(100).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['SMA300'] = df['Close'].rolling(300).mean()
            df['SMA10PC'] = df['SMA10'].pct_change()
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
            df.to_csv(fileName, index=False)
        except:
            print(f"Problem downloading: {tickerName}")
    else:
        print(f"File exists: {fileName}")

# Create required folders
makeRepo(stocksFolder)
makeRepo(imagesFolder)

# Create date and ticker-specific options and target prices folders
current_date = datetime.now().strftime('%Y-%m-%d')
optionsFolder = os.path.join(optionsBaseFolder, current_date, stockName)
targetPricesFolder = os.path.join(targetPricesBaseFolder, current_date, stockName)
makeRepo(optionsFolder)
makeRepo(targetPricesFolder)

# Download stock data
stockFile = os.path.join(stocksFolder, f"{stockName}_{myStart}_{myEnd}.csv")
downloadFile(stockName, stockFile, period)

# Calculate and save options data
generic = yf.Ticker(stockName)

if not generic.options:
    print("This guy is empty")
    sys.exit()

def calcPrice(atm, contractType, outputFile):
    atmStrike = atm['strike']
    atmBid = atm['bid']
    atmAsk = atm['ask']
    atmContract = atm['contractSymbol']
    atmAvg = round((atmBid + atmAsk) / 2, 2)
    atmActual = atmAvg * 100
    output = (
        f"{atmContract.values[0]} ::: {contractType} ATM Strike= {atmStrike.values[0]} BID PRICE TO OFFER: {atmAvg.values[0]} ACTUAL= ${round(atmActual.values[0], 2)}\n"
        f"  SELL @  [ 17% = {round(atmActual.values[0] * 1.17, 2)} ] [ 34% = {round(atmActual.values[0] * 1.34, 2)} ] [ 51% = {round(atmActual.values[0] * 1.51, 2)} ] [ 68% = {round(atmActual.values[0] * 1.68, 2)} ]\n"
        f"\t  [ 85% = {round(atmActual.values[0] * 1.85, 2)} ] [ 102% = {round(atmActual.values[0] * 2.02, 2)} ] [ 119% = {round(atmActual.values[0] * 2.19, 2)} ]\n"
    )
    print(output)
    with open(outputFile, 'a') as f:
        f.write(output)

def calcPrices(element):
    elementNumber = element
    option_chain = generic.option_chain(generic.options[elementNumber])
    calls = option_chain.calls
    puts = option_chain.puts
    
    currentDF = pd.concat([calls, puts])
    
    # Save to CSV
    optionsFile = os.path.join(optionsFolder, f"{stockName}_options_week_{elementNumber}.csv")
    currentDF.to_csv(optionsFile, index=False)
    print(f"Options data saved to {optionsFile}")

    # Save parsed output
    outputFile = os.path.join(targetPricesFolder, f"{stockName}_parsed_output_week_{elementNumber}.txt")
    
    dfCalls = currentDF[currentDF['contractSymbol'].str.match('.*C.*')]
    dfCalls = dfCalls[dfCalls['inTheMoney'] == False]
    atm = dfCalls.head(1)
    calcPrice(atm, "CALL", outputFile)

    dfPuts = currentDF[currentDF['contractSymbol'].str.match('.*P.*')]
    dfPuts = dfPuts[dfPuts['inTheMoney'] == False]
    atm = dfPuts.tail(1)
    calcPrice(atm, "PUT", outputFile)

print("Number weeks", numberWeeks)
for i in range(int(numberWeeks) + 1):
    print("Week=", i)
    calcPrices(i)

def makeTextPNG():
    sns.set_style("white")
    myTitle = "Options Pricing: " + str(stockName)
    myString = "Let's insert some stuff for stocks here:"
    
    fig2, ax2 = plt.subplots()
    ax2.set_title(myTitle, fontsize=22)
    ax2.text(0.0, 1.0, myString, fontsize=14, verticalalignment='top', color='navy')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.legend(loc='lower right')
    plt.savefig(imagesFolder + f'optionsPricing_{stockName}_{myStart}_{myEnd}.png')

# Uncomment to generate PNG
# makeTextPNG()

