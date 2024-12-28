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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200, figsize=(20, 10))
    
    # Rotated plot for calls (90 degrees to the right)
    ax1.plot(dfC['openInterest'] * dfC['lastPrice'], dfC['strike'], c='blue')
    ax1.scatter(dfC['openInterest'] * dfC['lastPrice'], dfC['strike'], s=(dfC['openInterest'] * dfC['lastPrice']) / myValue, alpha=.25, c="darkgreen")
    ax1.set_title(f'Calls: {title}')
    ax1.set_xlabel('Open Interest * Last Price')
    ax1.set_ylabel('Strike Price')
    ax1.set_xlim(left=0)  # Start x-axis at 0
    ax1.invert_xaxis()  # Invert the x-axis to have x=0 at the bottom
    
    # Rotated plot for puts (90 degrees to the left)
    ax2.plot(dfP['openInterest'] * dfP['lastPrice'], dfP['strike'], c='red')
    ax2.scatter(dfP['openInterest'] * dfP['lastPrice'], dfP['strike'], s=(dfP['openInterest'] * dfP['lastPrice']) / myValue, alpha=.10, c="navy")
    ax2.set_title(f'Puts: {title}')
    ax2.set_xlabel('Open Interest * Last Price')
    ax2.set_ylabel('Strike Price')
    ax2.set_ylim(bottom=0)  # Start y-axis at 0
    ax2.invert_yaxis()  # Invert the y-axis to have y=0 at the left
    
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

