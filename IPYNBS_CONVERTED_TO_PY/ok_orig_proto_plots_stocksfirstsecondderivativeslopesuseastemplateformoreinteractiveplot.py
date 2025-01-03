# -*- coding: utf-8 -*-
"""ok_orig_proto_plots_stocksFirstSecondDerivativeSlopesUseAsTemplateForMoreInteractivePlot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ANJMZqslyF2PQYjZNASUWUf21hULM54M
"""

#!pip install yfinance

import sys
equityType='Stock'
stockName='AMZN'
#equityType=sys.argv[1]
#stockName=sys.argv[2]
myStart='2021-01-01'
myEnd='2023-01-01'
sectorName="QQQ"

#equityType='Stock'

print("startDate:"+myStart)
print("endDate:"+myEnd)


import matplotlib.pyplot as plt
plt.clf()
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import datetime
import requests
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.clf()
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import openpyxl
import statsmodels.formula.api as smf
import statsmodels.formula.api as ols
from statsmodels.compat import lzip

#@title Functions for derivatives
def calcFirstDerivative(myDataframe):
  a=myDataframe[1:]
  b=myDataframe[:-1]
  a=a.reset_index()
  b=b.reset_index()
  c=b['Close']-a['Close']
  return c

def calcSecondDerivative(myDataframe):
  a=myDataframe[1:]
  b=myDataframe[:-1]
  a=a.reset_index()
  b=b.reset_index()
  c=b-a
  return c

import sys,os
#Variables
stocksFolder='./stocks/'
imagesFolder='./PNGS/'
pdfsFolder='./PDFS/'
period='1d'
stockFile=stocksFolder+stockName+'_'+myStart+'_'+myEnd+".csv"

def makeRepo(folder):
  if not os.path.isdir(folder):
    try:
      os.makedirs(folder)
      print("created folder")
    except:
      print("problem creating directory:",folder)
  else:
      print("directory exists!",folder)


def downloadFile(tickerName,fileName,period):
  if not os.path.isfile(fileName):
    try:
      print("download the file:",fileName)
      downloadDF=yf.download(tickerName,start=myStart,end=myEnd,period=period)
      df=downloadDF
      df=df.reset_index()
      #order matters
      df['SMA10']=df['Close'].rolling(10).mean()
      df['SMA30']=df['Close'].rolling(30).mean()
      df['SMA50']=df['Close'].rolling(50).mean()
      df['SMA100']=df['Close'].rolling(100).mean()
      df['SMA200']=df['Close'].rolling(200).mean()
      df['SMA300']=df['Close'].rolling(300).mean()
      df['SMA10PC']=df['SMA10'].pct_change()
      df['SMA30PC']=df['SMA30'].pct_change()
      df['SMA50PC']=df['SMA50'].pct_change()
      df['SMA100PC']=df['SMA100'].pct_change()
      df['SMA200PC']=df['SMA200'].pct_change()
      df['ClosePC']=df['Close'].pct_change()
      df['VolumePC']=df['Volume'].pct_change()
      df['VolLOG'] = np.log2(df['Volume'])
      df['CloseLOG'] = np.log2(df['Close'])
      df['VLOGPC'] = df['VolLOG'].pct_change()
      df['CLOGPC'] = df['CloseLOG'].pct_change()
      #first=calcFirstDerivative(df['Close'])
      #second=calcSecondDerivative(first)
      #df['FIRSTDERIV']=first
      #df['SECONDDERIV']=second
      df.to_csv(fileName)
    except:
      print("Problem downloading:",tickerName)
  else:
    print("file exits!",fileName)

#trigger it:
makeRepo(stocksFolder)
makeRepo(imagesFolder)
downloadFile(stockName,stockFile,period)



######### START FIRST DERIVATIVE ######

import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("white")
first=calcFirstDerivative(df['Close'])
second=calcSecondDerivative(first)
df['Close']=first
'''
print("last five periods: 1st Derivative:\n")
print(first.tail(5))
print("last five periods: 2nd Derivative:\n")
print(second['Close'].tail(5))

derivativeTail = "".join(df['Close'].astype('str').tail(5).tolist())
'''
def printMyStats(myRange):
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))

colName='Close'
#modified for the derivatives:
rp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='gold', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='lightblue', scatter_kws={'s':5})
xp = sns.lineplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='navy',linewidth=1)

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()


def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')

deviations=['.25','.5','.75','1.25','1.5','1.75','2.25','2.5','2.75','3.25','3.5','3.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4']
myColor='black'
drawLines(deviations,myColor)
colName='firstDerivative'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#plt.text(0.1, 0.9, 'text', size=15, color='purple')
derivString=str(df['Close'].tail(6).head(5))
#plt.text(0, 10,'last five periods:'+str(df['Close'].tail(5)), size=15, color='red')
plt.text(0, 7,'most recent slope values:\n'+derivString, size=15, color='red')
plt.title(str(colName).upper()+" "+stockName.upper()+" "+str(myStart)+" to: "+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'b_'+colName+'_'+stockName+'_'+myStart+'_'+myEnd+'.png')
#plt.savefig(imagesFolder+colName+'_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')
######### END FIRST DERIVATIVE ######

######### START SECOND DERIVATIVE ######
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("white")
first=calcFirstDerivative(df['Close'])
second=calcSecondDerivative(first)
df['Close']=second['Close']
'''
print("last five periods: 1st Derivative:\n")
print(first.tail(5))
print("last five periods: 2nd Derivative:\n")
print(second['Close'].tail(5))

derivativeTail = "".join(df['Close'].astype('str').tail(5).tolist())
'''
def printMyStats(myRange):
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))

colName='Close'
#modified for the derivatives:
rp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='red', scatter_kws={'s':75})
#zp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='red', scatter_kws={'s':5})
xp = sns.lineplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='green',linewidth=1)

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()


def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')

deviations=['.25','.5','.75','1.25','1.5','1.75','2.25','2.5','2.75','3.25','3.5','3.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4']
myColor='black'
drawLines(deviations,myColor)
colName='secondDerivative'
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#plt.text(0.1, 0.9, 'text', size=15, color='purple')
derivString=str(df['Close'].tail(6).head(5))
#plt.text(0, 10,'last five periods:'+str(df['Close'].tail(5)), size=15, color='red')
plt.text(0, 7,'most recent slope values:\n'+derivString, size=15, color='navy')
plt.title(str(colName).upper()+" "+stockName.upper()+" "+str(myStart)+" to: "+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'b_'+colName+'_'+stockName+'_'+myStart+'_'+myEnd+'.png')
######### END SECOND DERIVATIVE ######

import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("white")

fig,ax=plt.subplots(1,1)
CloseString=str(df['Close'].tail(6).head(5))
VolumeString=str(df['Volume'].tail(6).head(5))
DateString=str(df['Date'].tail(6).head(5))
OpenString=str(df['Open'].tail(6).head(5))
dailyVolatilityString=str(df['High'].tail(6).head(5)-df['Low'].tail(6).head(5))

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#rc = {'figure.figsize':(10,5),
#      'axes.facecolor':'white',
#      'axes.grid' : True,
#      'grid.color': '.8',
#      'font.size' : 15}
#plt.rcParams.update(rc)
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
ax.text(0, 1,'Last five Periods mini dataframe: '+stockName,size=12,color='black')
ax.text(0, 0,'last 5 closingPrice:\n'+CloseString, size=12, color='navy')
ax.text(.5, .5,'Trading Volume:\n'+VolumeString, size=12, color='green')
ax.text(0.55, 0,'OpeningPrice:\n'+OpenString, size=12, color='black')
ax.text(.22, .3,'Dates:\n'+DateString, size=12, color='Purple')
ax.text(0,.5,'DailyVolatility price range:\n'+dailyVolatilityString, size=12, color='red')
#plt.title(str(colName).upper()+" "+stockName.upper()+" "+str(myStart)+" to: "+str(myEnd))
#plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'x_miniDataframeStats_'+stockName+'_'+myStart+'_'+myEnd+'.png')
#print(df.tail())

import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("white")

fig,ax=plt.subplots(1,1)
a=str(df['SMA10'].tail(6).head(5))
b=str(df['SMA30'].tail(6).head(5))
c=str(df['SMA50'].tail(6).head(5))
d=str(df['SMA100'].tail(6).head(5))
e=str(df['SMA200'].tail(6).head(5)-df['Low'].tail(6).head(5))

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#rc = {'figure.figsize':(10,5),
#      'axes.facecolor':'white',
#      'axes.grid' : True,
#      'grid.color': '.8',
#      'font.size' : 15}
#plt.rcParams.update(rc)
ax.yaxis.set_visible(False)
ax.xaxis.set_visible(False)
ax.text(0, 1,'SMAs last 5 Periods aka current critical target prices : '+stockName,size=12,color='black')
ax.text(0, 0,'SMA10:\n'+a, size=12, color='navy')
ax.text(.5, .5,'SMA30:\n'+b, size=12, color='green')
ax.text(0.55, 0,'SMA50:\n'+c, size=12, color='black')
ax.text(.22, .3,'SMA100:\n'+d, size=12, color='Purple')
ax.text(0,.5,'SMA200:\n'+e, size=12, color='red')
#plt.title(str(colName).upper()+" "+stockName.upper()+" "+str(myStart)+" to: "+str(myEnd))
#plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'x_miniDataframeSMAs_'+stockName+'_'+myStart+'_'+myEnd+'.png')
print(df.tail())

import scipy.stats as scs
import statsmodels.api as sm
#sns.set_style("white")
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set(rc={'figure.figsize':(8,6)})
sns.set_style("white")
df=pd.read_csv(stockFile)
df = df[['Adj Close']].rename(columns={'Adj Close': 'adj_close'})
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))
df = df[['adj_close', 'log_rtn']].dropna(how = 'any')
r_range = np.linspace(min(df.log_rtn), max(df.log_rtn), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# histogram
sns.distplot(df.log_rtn, kde=False, norm_hist=True, ax=ax[0])
ax[0].set_title('Distribution of '+stockName+' returns', fontsize=16)
ax[0].plot(r_range, norm_pdf, 'g', lw=2,
           label=f'N({mu:.2f}, {sigma**2:.4f})')
ax[0].legend(loc='upper left');

# Q-Q plot
qq = sm.qqplot(df.log_rtn.values, line='s', ax=ax[1])
ax[1].set_title('Q-Q plot', fontsize = 16)
plt.savefig(imagesFolder+'distribution_and_qqplot_'+stockName+'_'+myStart+'_'+myEnd+'.png')

# plt.tight_layout()
# plt.savefig('images/ch1_im10.png')
plt.show()

#!rm -rf PNGS/
