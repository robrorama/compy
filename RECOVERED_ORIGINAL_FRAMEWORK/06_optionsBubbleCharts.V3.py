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
myStart = '2024-01-01'
myEnd = '2025-01-01'
maxValueFilter = 1000000
minValueFilter = 0

# Folders
imagesFolder = './PNGS/'

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

def makePlot(elementNumber, title):
    option_chain = generic.option_chain(generic.options[elementNumber])
    dfC = option_chain.calls
    dfP = option_chain.puts
    
    # CALCULATE FOR THE BUBBLE SIZES
    myMax = max(pd.concat([dfC['openInterest'] * dfC['lastPrice'], dfP['openInterest'] * dfP['lastPrice']]))
    oMag = 0 if np.isnan(myMax) or myMax == 0 else math.floor(math.log10(myMax))
    myValue = 10**(oMag - 3) if oMag > 3 else 1
    
    #plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=200, figsize=(10, 8))
    
    ax1.plot(dfC['strike'], dfC['openInterest'] * dfC['lastPrice'], c='blue')
    ax1.scatter(dfC['strike'], dfC['openInterest'] * dfC['lastPrice'], s=(dfC['openInterest'] * dfC['lastPrice']) / myValue, alpha=.25, c="darkgreen")
    ax1.set_title(f'Calls: {title}')
    
    ax2.plot(dfP['strike'], dfP['openInterest'] * dfP['lastPrice'], c='red')
    ax2.scatter(dfP['strike'], dfP['openInterest'] * dfP['lastPrice'], s=(dfP['openInterest'] * dfP['lastPrice']) / myValue, alpha=.10, c="navy")
    ax2.set_title(f'Puts: {title}')
    
    fig.tight_layout(pad=3.0)
    plt.savefig(os.path.join(imagesFolder, f'{title}_optionsContracts.png'))
    plt.show()

# Ensure directories exist
makeRepo(imagesFolder)

# Fetch data
generic = yf.Ticker(stockName)
if generic.options is None:
    print("No options available for this ticker.")
else:
    for i in range(len(generic.options)):
        title = f"{stockName.upper()}_{generic.options[i]}"
        makePlot(i, title)

