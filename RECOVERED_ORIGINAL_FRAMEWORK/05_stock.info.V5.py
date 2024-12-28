import sys
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt

# Variables
equityType = "Stock"
stockName = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
numberWeeks = int(sys.argv[2]) if len(sys.argv) > 2 else 4
myStart = '2024-01-01'
myEnd = '2025-01-01'
maxValueFilter = 1000000
minValueFilter = 0

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

def formatCompanyOfficers(officers):
    formatted_officers = "Company Officers\n"
    formatted_officers += "-" * 40 + "\n"
    for officer in officers:
        for key, value in officer.items():
            formatted_officers += f"{key.replace('_', ' ').capitalize():<30}: {value}\n"
        formatted_officers += "-" * 40 + "\n"
    return formatted_officers

def stockInfo(generic):
    info_dict = generic.info
    info_content = "Stock Information\n"
    info_content += "=" * 40 + "\n"
    for key, value in info_dict.items():
        if key == 'companyOfficers':
            info_content += formatCompanyOfficers(value)
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value)
            info_content += f"{key.replace('_', ' ').capitalize():<30}: {value}\n"
        else:
            info_content += f"{key.replace('_', ' ').capitalize():<30}: {value}\n"
    saveToFile(info_content, f"{stockName}_info.txt")
    print(info_content)

def calcPrice(atm, contractType):
    atmStrike = atm['strike'].values[0]
    atmBid = atm['bid'].values[0]
    atmAsk = atm['ask'].values[0]
    atmContract = atm['contractSymbol'].values[0]
    atmAvg = round((atmBid + atmAsk) / 2, 2)
    atmActual = atmAvg * 100
    price_info = (
        f"{atmContract} ::: {contractType} ATM Strike = {atmStrike}\n"
        f"BID PRICE TO OFFER: {atmAvg} ACTUAL = ${round(atmActual, 2)}\n"
        f"SELL @  [ 17% = {round(atmActual * 1.17, 2)} ] "
        f"[ 34% = {round(atmActual * 1.34, 2)} ] "
        f"[ 51% = {round(atmActual * 1.51, 2)} ] "
        f"[ 68% = {round(atmActual * 1.68, 2)} ]\n"
        f"[ 85% = {round(atmActual * 1.85, 2)} ] "
        f"[ 102% = {round(atmActual * 2.02, 2)} ] "
        f"[ 119% = {round(atmActual * 2.19, 2)} ]\n"
    )
    print(price_info)
    return price_info

def calcPrices(element):
    option_chain = generic.option_chain(generic.options[element])
    currentDF = pd.concat([option_chain.calls, option_chain.puts])
    dfCalls = currentDF[currentDF['contractSymbol'].str.contains('C')]
    atmCall = dfCalls[dfCalls['inTheMoney'] == False].head(1)
    call_info = calcPrice(atmCall, "CALL")

    dfPuts = currentDF[currentDF['contractSymbol'].str.contains('P')]
    atmPut = dfPuts[dfPuts['inTheMoney'] == False].tail(1)
    put_info = calcPrice(atmPut, "PUT")

    return call_info + put_info

def makeTextPNG():
    sns.set_style("white")
    fig2, ax2 = plt.subplots()
    ax2.set_title(f"Options Pricing: {stockName}", fontsize=22)
    ax2.text(0.0, 1.0, "Let's insert some stuff for stocks here:", fontsize=14, verticalalignment='top', color='navy')
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    ax2.legend(loc='lower right')
    plt.savefig(os.path.join(imagesFolder, f'optionsPricing_{stockName}_{myStart}_{myEnd}.png'))

# Ensure directories exist
makeRepo(stocksFolder)
makeRepo(imagesFolder)
makeRepo(txtFolder)

# Fetch data
generic = yf.Ticker(stockName)
if generic.options is None:
    print("No options available for this ticker.")
    sys.exit()

# Get and save stock info
stockInfo(generic)

# Calculate and save prices for options
options_info = f"Number of weeks: {numberWeeks}\n"
options_info += "=" * 40 + "\n"
for i in range(numberWeeks + 1):
    options_info += f"Week = {i}\n"
    options_info += calcPrices(i)
    options_info += "-" * 40 + "\n"

saveToFile(options_info, f"{stockName}_optionsOut.txt")

# Generate PNG with text
makeTextPNG()

print("exiting")
sys.exit()

