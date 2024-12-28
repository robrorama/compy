# this is the same script just the matrix generator the rest commented out

import sys
equityType=sys.argv[1] 
stockName=sys.argv[2] 
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
  
#first=calcFirstDerivative(df['Close'])
#second=calcSecondDerivative(first)

#@title  revamped Local data download : 
import sys,os
#Variables
stocksFolder='./stocks/'
imagesFolder='./PNGS/'
pdfsFolder='./PDFS/'
period='1d'

stockFile=stocksFolder+stockName+'_'+myStart+'_'+myEnd+".csv"
#sectorFile=stocksFolder+sectorName+'_'+myStart+'_'+myEnd+".csv"
qqqFile=stocksFolder+'qqq_'+myStart+'_'+myEnd+".csv"
goldFile=stocksFolder+'gold_'+myStart+'_'+myEnd+".csv"
spyFile=stocksFolder+'spy_'+myStart+'_'+myEnd+".csv"
iwmFile=stocksFolder+'iwm_'+myStart+'_'+myEnd+".csv"
eemFile=stocksFolder+'eem_'+myStart+'_'+myEnd+".csv"
tltFile=stocksFolder+'tlt_'+myStart+'_'+myEnd+".csv"
diaFile=stocksFolder+'dia_'+myStart+'_'+myEnd+".csv"
btcFile=stocksFolder+'btc_'+myStart+'_'+myEnd+".csv"
ungFile=stocksFolder+'ung_'+myStart+'_'+myEnd+".csv"
uupFile=stocksFolder+'uup_'+myStart+'_'+myEnd+".csv"
fxbFile=stocksFolder+'fxb_'+myStart+'_'+myEnd+".csv"
eurFile=stocksFolder+'eur_'+myStart+'_'+myEnd+".csv"
usoFile=stocksFolder+'uso_'+myStart+'_'+myEnd+".csv"
fxiFile=stocksFolder+'fxi_'+myStart+'_'+myEnd+".csv"
slvFile=stocksFolder+'slv_'+myStart+'_'+myEnd+".csv"
kreFile=stocksFolder+'kre_'+myStart+'_'+myEnd+".csv"
xlfFile=stocksFolder+'xlf_'+myStart+'_'+myEnd+".csv"
xlbFile=stocksFolder+'xlb_'+myStart+'_'+myEnd+".csv"
xmeFile=stocksFolder+'xme_'+myStart+'_'+myEnd+".csv"
xlpFile=stocksFolder+'xlp_'+myStart+'_'+myEnd+".csv"
xluFile=stocksFolder+'xlu_'+myStart+'_'+myEnd+".csv"
xliFile=stocksFolder+'xli_'+myStart+'_'+myEnd+".csv"
ibbFile=stocksFolder+'ibb_'+myStart+'_'+myEnd+".csv"
xbiFile=stocksFolder+'xbi_'+myStart+'_'+myEnd+".csv"
xlvFile=stocksFolder+'xlv_'+myStart+'_'+myEnd+".csv"
xlyFile=stocksFolder+'xly_'+myStart+'_'+myEnd+".csv"
xrtFile=stocksFolder+'xrt_'+myStart+'_'+myEnd+".csv"
xhbFile=stocksFolder+'xhb_'+myStart+'_'+myEnd+".csv"
xleFile=stocksFolder+'xle_'+myStart+'_'+myEnd+".csv"
xopFile=stocksFolder+'xop_'+myStart+'_'+myEnd+".csv"
xlkFile=stocksFolder+'xlk_'+myStart+'_'+myEnd+".csv"




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
#downloadFile(sectorName,sectorFile,period)
downloadFile('qqq',qqqFile,period)
downloadFile('gold',goldFile,period)
downloadFile('spy',spyFile,period)
downloadFile('iwm',iwmFile,period)
downloadFile('eem',eemFile,period)
downloadFile('tlt',tltFile,period)
downloadFile('dia',diaFile,period)
downloadFile('btc-usd',btcFile,period)
downloadFile('ung',ungFile,period)
downloadFile('uup',uupFile,period)
downloadFile('fxb',fxbFile,period)
downloadFile('eur',eurFile,period)
downloadFile('uso',usoFile,period)
downloadFile('fxi',fxiFile,period)
downloadFile('slv',slvFile,period)
downloadFile('kre',kreFile,period)
downloadFile('xlf',xlfFile,period)
downloadFile('xlb',xlbFile,period)
downloadFile('xle',xleFile,period)
downloadFile('xlp',xlpFile,period)
downloadFile('xlu',xluFile,period)
downloadFile('xli',xliFile,period)
downloadFile('ibb',ibbFile,period)
downloadFile('xlv',xlvFile,period)
downloadFile('xly',xlyFile,period)
downloadFile('xrt',xrtFile,period)
downloadFile('xhb',xhbFile,period)
downloadFile('xle',xleFile,period)
downloadFile('xop',xopFile,period)
downloadFile('xlk',xlkFile,period)


#### START MAKING CORRELATION MATRIX #####
#printMyStats(df['Close'])
def makeCorrelationMatrix():
    dfslist = [f for f in os.listdir(stocksFolder) if os.path.isfile( os.path.join(stocksFolder, f) )]
    return dfslist


myCorrDf=pd.DataFrame()
#count=0
for i in makeCorrelationMatrix():
    foo=stocksFolder+i
    #print(foo)
    test=i.split('_')
    #print(test[0])
    a=pd.read_csv(foo)
    #print(a.head())
    myCorrDf[test[0]]=a['CloseLOG']
    #myCorrDf[test[0]]=a['Close']
    #count=count+1

#fig,ax = plt.subplots()
fig,ax = plt.subplots(figsize=(8,6))
#plt.rc("figure", figsize=(10,8),dpi=300)
cmap=sns.color_palette("Blues",as_cmap=True)
sns.heatmap(data=myCorrDf.corr(),##corr matrix
        linewidth=0,#lines sep squares
        square=True,# 1:1 cell ratio
        cmap=cmap,#colormap
        vmax=1,vmin=-1,center=0,#range
        cbar_kws={"shrink":.50} #shrink scale
        )
#plt.ticks(rotation=0)
plt.title(stockName+' Start Date'+myStart+' EndDate:'+myEnd)
matrixImage=stockName+'b_'+myStart+'_'+myEnd+'_matrix.png'
plt.savefig(imagesFolder+matrixImage,bbox_inches='tight')



#exit debugging
print("exiting")
sys.exit()



print(" AM I GETTING THIS FAR BECAUSE IF I AM IT IS A MISTAKE !")
##### this is the main LRC graph #######
#@title make the image : maybe even remove this and just change the input into the function in the next block 
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
plt.rc("figure", figsize=(10,8),dpi=300)
plt.rc("font", size=14) 

print(df.tail())

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#only df where > 2*np.std(...?)
rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='lightblue', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='navy', scatter_kws={'s':5})

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()

def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
   
deviations=['.25','.5','.75','1','1.25','1.5','1.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4','5']
myColor='black'
drawLines(deviations,myColor)

smas=['SMA10','SMA30','SMA50','SMA100','SMA200','SMA300']
smaColors=['lightblue','g','orange','blue','purple','red']
for i in range(len(smas)):
  sns.lineplot(x=df.index,y=smas[i],data=df,color=smaColors[i],linewidth=1.6)

#sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#print("derbyderby")
#plt.rc("figure", figsize=(1,1),dpi=300)
#plt.rc("font", size=14) 
plt.title("LRC:  "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+stockName+'_'+myStart+'_'+myEnd+'.png')
#plt.show()
##### END main LRC graph #######



#@title : define lines to be drawn and colors This one has to happen 

#  smas=['SMA30PC','SMA50PC','SMA100PC','SMA200PC','ClosePC','VolumePC']
#  smaColors=['g','orange','blue','purple','red','yellow']  
if len(df)>201:
  myLines=['SMA30PC','SMA50PC','SMA100PC','SMA200PC']
  myColors=['g','orange','blue','purple']  
  print("over 201")
elif len(df)>101:
  myLines=['SMA30PC','SMA50PC','SMA100PC']
  myColors=['g','orange','blue']
  print("over 201")
elif len(df)>51:
  myLines=['SMA30PC','SMA50PC']
  myColors=['g','orange']
  print("over 51")
elif len(df)>31:
  myLines=['SMA30PC']
  myColors=['g']  
  print("over 31")


#### START SMA GRAPHS #### 
###@title see percentage graphs 
####@title : plot regression channel 
import matplotlib.pyplot as plt
import pandas as pd
plt.clf()
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#myRange='SMA30PC'
myRange=myLines[0]

#only df where > 2*np.std(...?)
rp = sns.regplot(x=df.index, y=myRange, data=df, ci=None, marker='.', color='lawngreen', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y=myRange, data=df, ci=None, marker='.', color='black', scatter_kws={'s':5})

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()

def printMyStats(myRange):
  #mystd=df['SMA30PC'].std()
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  #mymean=df['SMA30PC'].mean()
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))

printMyStats(myRange)

def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
  
deviations=['.25','.5','.75','1','1.25','1.5','1.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4','5']
myColor='black'
drawLines(deviations,myColor)

for i in range(len(smas)-1):
  sns.lineplot(x=df.index,y=myLines[i],data=df,color=myColors[i],linewidth=.6)

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.title("LRC: percentage graphs "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'movingAverages_percentages_'+stockName+'_'+myStart+'_'+myEnd+'.png')

#plt.show()

### END SMA GRAPH #####

def printMyStats(myRange):
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))

printMyStats(myRange)



#### START VOLUME GRAPH ####
print("processing volume graph ")
###@title another try just for volume : 
#myLines=['VLOGPC']
#myColors=['gold']  
import matplotlib.pyplot as plt
import pandas as pd
plt.clf()
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
myRange=myLines[0]



colName='VolLOG'
rp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='lawngreen', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='black', scatter_kws={'s':5})
xp = sns.lineplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='gold',linewidth=.4)

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()


def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
  
deviations=['.25','.5','.75','1','1.25','1.5','1.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4','5','10','20']
myColor='black'
drawLines(deviations,myColor)

#for i in range(len(smas)-1):
###sns.lineplot(x=df.index,y=myLines[0],data=df,color=myColors[0],linewidth=.6)

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.title("LRC: Volume LOG graphs "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+'volumeLOG_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')

#plt.show()
#### END VOLUME GRAPH ####

print("END VOLUME GRAPH")




'''
###@title another try just for volume : 
#myLines=['VLOGPC']
#myColors=['gold']  
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#myRange=myLines[0]

def printMyStats(myRange):
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))
colName='Close'
rp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='gold', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='brown', scatter_kws={'s':5})
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

#for i in range(len(smas)-1):
###sns.lineplot(x=df.index,y=myLines[0],data=df,color=myColors[0],linewidth=.6)

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.title("LRC: "+colName+" graphs "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+colName+'_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')

#plt.show()


printMyStats(myRange)
'''



# Commented out IPython magic to ensure Python compatibility.
def indentify_outliers(row, n_sigmas=3):
    '''
    # originally from the python for finance cookbook 
    Function for identifying the outliers using the 3 sigma rule. 
    The row must contain the following columns/indices: simple_rtn, mean, std.
    
    Parameters
    ----------
    row : pd.Series
        A row of a pd.DataFrame, over which the function can be applied.
    n_sigmas : int
        The number of standard deviations above/below the mean - used for detecting outliers
        
    Returns
    -------
    0/1 : int
        An integer with 1 indicating an outlier and 0 otherwise.
    '''
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1
    else:
        return 0 

import pandas as pd 
import yfinance as yf
import math,sys,warnings
#import yfinance as yf
# %matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
plt.clf()
import numpy as np
import seaborn as sns 
plt.style.use('seaborn')
# plt.style.use('seaborn-colorblind') #alternative
# plt.rcParams['figure.figsize'] = [16, 9]
###plt.rcParams['figure.dpi'] = 100
warnings.simplefilter(action='ignore', category=FutureWarning)
df=pd.read_csv(stockFile)
#stock='TQQQ'
#df = yf.download(stock, 
#                 start='2020-01-01', 
#                 #start='2020-12-1',
#                 end='2021-12-31',
#                 progress=False)
#massage dataframe

df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)
df['simple_rtn'] = df.adj_close.pct_change()
df.head()
df_rolling = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
df_rolling.columns = df_rolling.columns.droplevel()
df_outliers = df.join(df_rolling)
df_outliers['outlier'] = df_outliers.apply(indentify_outliers,axis=1)
outliers = df_outliers.loc[df_outliers['outlier'] == 1,['simple_rtn']]

print("finished doing outliers")


def showThreeSD():
  fig, ax = plt.subplots()
  ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
  ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
  ax.set_title(stockName+" 2020 (3xstd.dev) stock price moves this period==>"+ str(outliers.simple_rtn.count()))
  ax.legend(loc='lower right')
  #graph of outliers :
  plt.savefig(imagesFolder+'3_standard_deviations_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')
#  plt.show(fig)
  #plt.show()

showThreeSD()


## commented out come fix later this one 

#df.to_csv('testout2.csv')
#!cp testout2.csv "/content/drive/"

## consecutive up down days : 
# i want to take the simple return and see . convert to how many concurrent days up or down
df2=df['simple_rtn'].copy(deep=True)
df2[df2>0]=1
df2[df2<0]=-1
df3=df2.copy(deep=True)
#ax2.plot(df['simple_rtn'],color='green', label='Normal')

#old=0
tally=1
old=1
runningSum=0
for value in df3:
  if ( math.isnan(value)):
      print(" skipping a nan ")
  else:
    combined=int(old)+int(value)
    if( combined == 0 ) :
      mesg="PN"
      #print("one poz one neg ")
      runningSum=0
    elif( combined < 0 ) : 
      mesg="2N"
      #print ( " two negs ")
      runningSum=runningSum-1
    elif ( combined > 0 ):
      mesg="2P"
      #print ( " two poz")
      runningSum=runningSum+1
    tally=tally+1
    old=value
    #for debugging
    ##print("check : "+mesg+" iteration:"+str(tally)+" runningSum:"+str(runningSum))
    #print(df3.head())
    ## This is a modification to see what happens when I zero anything under 2 ? and above -2 ?
    #   it does not seem to work yet 
    #if(runningSum<=2 & runningSum>=-2):
    
    df3.iloc[tally-1]=runningSum

def showUpDownDays():
  #do the stats for the history of the stock
  totalDays=df3.count()
  print("UP/Down Consecutive Trading Days out of a Total of :"+str(totalDays)+" Days for StockName:"+stockName)
  for i in range(0,int(df3.max()),1):
    myDays=df3[(df3==i)|(df3==-i)].count()
    print(str(i)+"   [[  "+str(myDays)+"  ]]   ,  "+str((100*((myDays/totalDays))).round(2))+"%")
  #print("Total trading days:",df3.count())
  #print("Up/Down 1 day",df3[(df3==1)|(df3==-1)].count())
  print(" ************************************************* ")
  print("Last five trading days:",df3.tail(5))
  print("come back later and turn this guy into a data frame ")
## show the text up down 
showUpDownDays()

def showConsecutiveDays():
  #do zeroing out ?   
  ###df3[(df3<5)&(df3>-5)]=0
  df3[(df3<3)&(df3>-3)]=0
  fig2, ax2 = plt.subplots()
  ax2.plot(df3,color='green', label='Normal')
  #ax2.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
  ax2.set_title(stockName+" simple return :"+ str(stockName))
  ax2.legend(loc='lower right')
  plt.savefig(imagesFolder+'consecutive_up_down_days_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')

  #plt.show()
  #plt.show(fig2)


  ##tally some stats : 5 days up, 6 days up , 7 days up, 8 days up, over 8 
  ##five=df3[df3==5].count()
  ##print("tally of five consecutive days : "+str(five))
  #print("consecutive days last two trading days: ",df3.tail(2))
  #print("consecutive days up the previous trading day",df3.tail(:,-1))

## make this one work later 
showConsecutiveDays()



def makePieChart():
  generic=yf.Ticker(stockName)
  analystRatings=generic.recommendations['To Grade'].value_counts()
  #fig = plt.figure(figsize=(4,4), dpi=200)
  ax = plt.subplot(111)
  analystRatings.plot(kind='pie',title=stockName,ax=ax,autopct='%1.1f%%',startangle=270,fontsize=8)
  plt.savefig(imagesFolder+'pie_chart_ratings'+stockName+'_'+myStart+'_'+myEnd+'.png')
  #plt.show()

if equityType == 'Stock':
  print("equity Type is a stock ")
  print("not making PieChart")
  makePieChart()
else:
  print(" equityType is ETF : Doing nothing")







import scipy.stats as scs
import statsmodels.api as sm
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
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
#plt.show()

### This no works either 
'''

### try again for the correlations : 
#df = yf.download(['^GSPC', '^VIX'], 
#                 #year="2018"
#                 start='1985-01-01', 
##                 #start='2018-01-01',
#                 end='2018-12-31',
#                 progress=False)

df=pd.read_csv(stockFile)
df2=pd.read_csv(goldFile)


#df2=pd.read_csv(qqqFile)
#df = df[['Adj Close']]
#df.columns = df.columns.droplevel(0)
#df = df.rename(columns={'^GSPC': 'sp500', '^VIX': 'vix'})

df['log_rtn'] = np.log(df.Close / df.Close.shift(1))
df['vol_rtn'] = np.log(df2.Close / df2.Close.shift(1))

#df['vol_rtn'] = np.log(df.vix / df.vix.shift(1))
df.dropna(how='any', axis=0, inplace=True)

corr_coeff = df.log_rtn.corr(df.vol_rtn)

ax = sns.regplot(x='log_rtn', y='vol_rtn', data=df, 
                 line_kws={'color': 'red'})

ax.set(title=f'ONE VS ANOTHER ($\\rho$ = {corr_coeff:.2f})',
       ylabel='ylabel',
       xlabel='xlabel'+myStart+" "+myEnd)

#ax.set(title=f'S&P 500 vs. VIX ($\\rho$ = {corr_coeff:.2f})',
#       ylabel='VIX log returns',
#       xlabel='S&P 500 log returns Just '+myStart+" "+myEnd)
      

# plt.tight_layout()
# plt.savefig('images/ch1_im16.png')
plt.show()
'''

### THIS ONE NO WORKS : 
'''
def compareSymbols( symbol1,symbol2,comparisonName ):
  sns.set_style("white")
  sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
  print(symbol1)
  print(symbol2)
  a=pd.read_csv(symbol1)
  #print(a.tail())
  b=pd.read_csv(symbol2)
  #print(b.tail())
  newdf=pd.DataFrame()
  newdf['a']=a['Close']
  #newdf['a']=a['Close']
  newdf['b']=b['Close']
  newdf['log_rtn'] = np.log(newdf['a'] / newdf['a'].shift(1))
  newdf['vol_rtn'] = np.log(newdf['b'] / newdf['b'].shift(1))
 
  #newdf['log_rtn'] = np.log(newdf.a / newdf.a.shift(1))
  #newdf['vol_rtn'] = np.log(newdf.b / newdf.b.shift(1))
  print(newdf.tail())
  newdf.dropna(how='any', axis=0, inplace=True)
  corr_coeff = newdf.log_rtn.corr(newdf.vol_rtn)
  mytitle='Compare correlation '+symbol1+'\n vs.  \n '+comparisonName+' '
  ax = sns.regplot(x='log_rtn', y='vol_rtn', data=newdf,line_kws={'color': 'red'})
  ax.set(title=mytitle+f'( $\\rho$ = {corr_coeff:.2f})')
  plt.savefig(imagesFolder+'correlation_'+stockName+'_vs_'+comparisonName+'.png')
  newdf=pd.DataFrame()
  plt.show()

compareSymbols(qqqFile,stockFile,"QQQ")
compareSymbols(tltFile,stockFile,"TLT")
compareSymbols(spyFile,stockFile,"SPY")
compareSymbols(eemFile,stockFile,"EEM")
compareSymbols(iwmFile,stockFile,"IWM")
compareSymbols(diaFile,stockFile,"DIA")
compareSymbols(goldFile,stockFile,"GOLD")
'''

###@title another try just for volume : 
#myLines=['VLOGPC']
#myColors=['gold']  
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv(stockFile)
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
#myRange=myLines[0]
first=calcFirstDerivative(df['Close'])
second=calcSecondDerivative(first)
df['Close']=first
print(first.head())
print(second.head())

def printMyStats(myRange):
  mystd=df[myRange].std()
  print('StandardDeviation ',myRange,' PercentageChange:',mystd)
  mymean=df[myRange].mean()
  print('Mean ',myRange,' PercentageChange:',mymean)
  print(df.tail(5))

colName='Close'
#modified for the derivatives:
rp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='gold', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y=df[colName], data=df, ci=None, marker='.', color='brown', scatter_kws={'s':5})
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

#for i in range(len(smas)-1):
###sns.lineplot(x=df.index,y=myLines[0],data=df,color=myColors[0],linewidth=.6)
colName='first derivative'
##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.title("FIRST DERIVATIVE GRAPH: "+colName+" "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+colName+'_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')

#plt.show()


#### END MAKING CORRELATION MATRIX #####


###sys.exit()
#stocksFolder='./stocks/'



#make pdfs : 
#0,0,210,297
import os
from PIL import Image
from PIL.ExifTags import TAGS
path=imagesFolder
from fpdf import FPDF
pdf = FPDF()
#imagelist = [item for item in items if isfile(join(imagesFolder, item))]
imagelist = [f for f in os.listdir(path) if os.path.isfile( os.path.join(path, f) )]
# imagelist is the list with all image filenames
pdf = FPDF("L", unit = "pt", format = "legal")
for image in imagelist:
    imagename=path+image
    print("processing:"+imagename)
    #image = Image.open(imagename)
    #h=image.height
    #w=image.width
    #h=2400
    #w=4800
    #print("Image HxW=",h,w)
    pdf.add_page()
    pdf.image(imagename, 0, 0, 0, 525)
    #pdf.add_page()
    #pdf.image(path+image)
    #pdf.image(imagename,0,0,h,w)
    #pdf.image(image,x,y,w,h)
pdf.output(pdfsFolder+stockName+"_"+myStart+"_"+myEnd+".pdf", "F")
print("finished creating pdf")

print("FINISHED GENERATING PDF ")
print("EXITING")
sys.exit()

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.title("LRC: "+colName+" graphs "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.savefig(imagesFolder+colName+'_graph_'+stockName+'_'+myStart+'_'+myEnd+'.png')

#plt.show()

first=calcFirstDerivative(df['Close'])
second=calcSecondDerivative(first)
print("first derivative:")
print(first.tail())
print("second derivative:")
print(second.tail())

###@title see percentage graphs  This is the functionized version :
def fullThing(graphName,myLines,myColors,myRange):

  def printMyStats(myRange):
    mystd=df[myRange].std()
    print('StandardDeviation ',myRange,' PercentageChange:',mystd)
    mymean=df[myRange].mean()
    print('Mean ',myRange,' PercentageChange:',mymean)
    print(df.tail(5))


  def drawLines(deviations,color,x_rp,y_rp):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    for i in deviations:
      sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
      sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')

  def plotLRC(myLines,myColors,myRange):
    import matplotlib.pyplot as plt
    plt.clf()
    import pandas as pd
    rp = sns.regplot(x=df.index, y=myRange, data=df, ci=None, marker='.', color='lawngreen', scatter_kws={'s':75})
    zp = sns.regplot(x=df.index, y=myRange, data=df, ci=None, marker='.', color='black', scatter_kws={'s':5})
    y_rp = rp.get_lines()[0].get_ydata()
    x_rp = rp.get_lines()[0].get_xdata()
    z_rp = rp.get_lines()[0].get_ydata()
    deviations=['.25','.5','.75','1','1.25','1.5','1.75']
    myColor='lightgrey'
    drawLines(deviations,myColor,x_rp,y_rp)
    deviations=['1','2','3','4','5']
    myColor='black'
    drawLines(deviations,myColor,x_rp,y_rp)


 
  #Trigger the lines
  plotLRC(myLines,myColors,myRange)

  '''
  deviations=['.25','.5','.75','1','1.25','1.5','1.75']
  myColor='lightgrey'
  drawLines(deviations,myColor)
  deviations=['1','2','3','4','5']
  myColor='black'
  drawLines(deviations,myColor)
  '''
  for i in range(len(myLines)):
    sns.lineplot(x=df.index,y=myLines[i],data=df,color=myColors[i],linewidth=.6)

  ##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
  sns.set_style("white")
  sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
  plt.title("LRC: "+graphName+"  "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))
  plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
  plt.savefig(imagesFolder+graphName+'_'+stockName+'_'+myStart+'_'+myEnd+'.png')

  #plt.show()

  printMyStats(myRange)
#main
#percentage change SMAs graph 
#graphName='percentChange_movingAverages'
#fullThing(graphName,myLines,myColors)
#graphName='percentChange_closingPrice'
# close percentage change graph 
myLines=['ClosePC']
myColors=['lightblue']  
myRange=myLines[0]
fullThing(graphName,myLines,myColors,myRange)

## should run through and do a separate one for Volume and price close?

print("system exit")
sys.exit()

###@title : download data 
import sys,os
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=14)

outFolder='./stocks/'
myPath=outFolder
if not os.path.isdir(myPath):
  try:
    os.makedirs(myPath)
  except:
    print("problem creating directory:",myPath)

#print if i am going to check it lives there
# i need to download into files and reimport into pandas from there
from datetime import datetime
#print("started---->"+str(datetime.now()))
startTS=datetime.timestamp(datetime.now())
#print("timestamp--->"+str(startTS))
dfStock=yf.download(stockName,start=myStart,end=myEnd,period='1d')
dfSector=yf.download(sectorName,start=myStart,end=myEnd,period='1d')
#df=dfb/dfa
###df=dfStock/dfSector
df=dfStock
### now make the part that is the graph part into a function to reuse 
####df=dfa
#print(df.head())
df=df.reset_index()
#print(df.head())

#print("Finished---->"+str(datetime.now()))
endTS=datetime.timestamp(datetime.now())
#print("total RunTime--->"+str(endTS-startTS))

#@title : download data  ORIGINAL 
'''
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=14)

from datetime import datetime
#print("started---->"+str(datetime.now()))
startTS=datetime.timestamp(datetime.now())
#print("timestamp--->"+str(startTS))
df=yf.download(stockName,start=myStart,end=myEnd,period='1d')

#print(df.head())
df=df.reset_index()
#print(df.head())

#print("Finished---->"+str(datetime.now()))
endTS=datetime.timestamp(datetime.now())
#print("total RunTime--->"+str(endTS-startTS))
'''

####@title : plot regression channel 
import matplotlib.pyplot as plt
import pandas as pd
plt.clf()
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#only df where > 2*np.std(...?)
rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='lightblue', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='.', color='navy', scatter_kws={'s':5})

#####usingi numpy works 
#####rp = sns.regplot(x=X1, y=Y1, ci=None, color='green')

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()
 
print("STDDEV:")
mystd=df['Close'].std()
print("MEAN:")
mymean=df['Close'].mean()
print("##### data ####")
print(mymean)
print(" std: ",mystd," mean:",mymean)
print(df['Close'][df['Close']>df['Close'].std()].count())
print(df.tail(5))
#deviations=['.5','1','1.25','1.5','1.75','2','3','4']
#devColors=['grey','grey','grey','grey','blue','green','red']
#count=0
def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    #sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color='lightgrey',linewidth=.5,linestyle='-')
    #sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color='lightgrey',linewidth=.5,linestyle='-')

deviations=['.25','.5','.75','1','1.25','1.5','1.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4','5']
myColor='black'
drawLines(deviations,myColor)


df['SMA30']=df['Close'].rolling(30).mean()
df['SMA50']=df['Close'].rolling(50).mean()
df['SMA100']=df['Close'].rolling(100).mean()
df['SMA200']=df['Close'].rolling(200).mean()
df['SMA300']=df['Close'].rolling(300).mean()
smas=['SMA30','SMA50','SMA100','SMA200','SMA300']
smaColors=['g','orange','blue','purple','red']
for i in range(len(smas)):
  sns.lineplot(x=df.index,y=smas[i],data=df,color=smaColors[i],linewidth=1.6)

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#plt.rcParams['figure.dpi'] = 300
#plt.rcParams['savefig.dpi'] = 300
#plt.legend(handles=smas) 
#[eight, nine])
#@plt.legend(title='Simple Moving Averages  ', loc='center left', labels=['30','60','100','200'],bbox_to_anchor=(.5, -.2),
#          fancybox=True, shadow=True, ncol=5,color=smaColors)
###plt.title("LRC:  "+stockName.upper()+" relative to QQQ    from:"+str(myStart)+"   to:"+str(myEnd))
plt.title("LRC:  "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))


plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
###plt.yaxis.tick_right()
#plt.title("LRC: "+stockName.upper()+"   from:"+str(myStart)+"   to:"+str(myEnd))
#plt.savefig(stockName+'relative_to_qqq_'+myStart+'_'+myEnd+'.png')
plt.savefig(stockName+myStart+'_'+myEnd+'.png')

#plt.show()

#change all of the values that are being plotted into percentage change, then you can show them all together
#Close Volume SMA30 50 100 200 
df['SMA30PC']=df['SMA30'].pct_change()
df['SMA50PC']=df['SMA50'].pct_change()
df['SMA100PC']=df['SMA100'].pct_change()
df['ClosePC']=df['Close'].pct_change()
df['VolumePC']=df['Volume'].pct_change()
print("need to add in the graphics here ")

'''
print(df['SMA30PC'].tail(5))
print(df['SMA50PC'].tail(5))
print(df['SMA100PC'].tail(5))
print(df['SMA200PC'].tail(5))
print(df['ClosePC'].tail(5))
print(df['VolumePC'].tail(5))
'''

####@title : plot regression channel 
import matplotlib.pyplot as plt
import pandas as pd
plt.clf()
sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
'''
#only df where > 2*np.std(...?)
rp = sns.regplot(x=df.index, y='ClosePC', data=df, ci=None, marker='.', color='lightblue', scatter_kws={'s':75})
zp = sns.regplot(x=df.index, y='ClosePC', data=df, ci=None, marker='.', color='navy', scatter_kws={'s':5})

#####usingi numpy works 
#####rp = sns.regplot(x=X1, y=Y1, ci=None, color='green')

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()
 '''
print("STDDEV:")
mystd=df['Close'].std()
print("MEAN:")
mymean=df['ClosePC'].mean()
print("##### data ####")
print(mymean)
print(" std: ",mystd," mean:",mymean)
#print(df['Close'][df['Close']>df['Close'].std()].count())
print(df.tail(5))
#deviations=['.5','1','1.25','1.5','1.75','2','3','4']
#devColors=['grey','grey','grey','grey','blue','green','red']
#count=0

def drawLines(deviations,color):
  for i in deviations:
    sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color=myColor,linewidth=.5,linestyle='-')
    #sns.lineplot(x=x_rp, y=y_rp + ( float(i)* np.std(y_rp)), color='lightgrey',linewidth=.5,linestyle='-')
    #sns.lineplot(x=x_rp, y=y_rp - (float(i)* np.std(y_rp)), color='lightgrey',linewidth=.5,linestyle='-')

deviations=['.25','.5','.75','1','1.25','1.5','1.75']
myColor='lightgrey'
drawLines(deviations,myColor)
deviations=['1','2','3','4','5']
myColor='black'
drawLines(deviations,myColor)


#df['SMA30']=df['Close'].rolling(30).mean()
#df['SMA50']=df['Close'].rolling(50).mean()
#df['SMA100']=df['Close'].rolling(100).mean()
#df['SMA200']=df['Close'].rolling(200).mean()

#smas=['SMA30PC','SMA50PC','SMA100PC','SMA200PC','VolumePC']
#smaColors=['g','orange','blue','purple','grey']
smas=['SMA30PC','SMA50PC','SMA100PC','SMA200PC']
smaColors=['g','orange','blue','purple']

for i in range(len(smas)):
  sns.lineplot(x=df.index,y=smas[i],data=df,color=smaColors[i],linewidth=1.6)

##sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
#plt.rcParams['figure.dpi'] = 300
#plt.rcParams['savefig.dpi'] = 300
#plt.legend(handles=smas) 
#[eight, nine])
#@plt.legend(title='Simple Moving Averages  ', loc='center left', labels=['30','60','100','200'],bbox_to_anchor=(.5, -.2),
#          fancybox=True, shadow=True, ncol=5,color=smaColors)
###plt.title("LRC:  "+stockName.upper()+" relative to QQQ    from:"+str(myStart)+"   to:"+str(myEnd))
plt.title("LRC:  "+stockName.upper()+str(myStart)+"   to:"+str(myEnd))


plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
###plt.yaxis.tick_right()
#plt.title("LRC: "+stockName.upper()+"   from:"+str(myStart)+"   to:"+str(myEnd))
#plt.savefig(stockName+'relative_to_qqq_'+myStart+'_'+myEnd+'.png')
plt.savefig(stockName+myStart+'_'+myEnd+'.png')

#plt.show()

print("EXIT HERE ")
import os,sys
sys.exit()

#@title : manipulate data frame 
df['percentChange']=df['Close'].pct_change()
df['above']=df['Close']
df['below']=df['Close']
#df['below'][df['below']>(np.mean(df['pctChange'])-1.5*np.std(df['pctChange']))]=0
#df['above'][df['above']<(np.mean(df['pctChange'])+1.5*np.std(df['pctChange']))]=0
df['below'][df['below']>(np.mean(df['Close'])- 1.0*np.std(df['Close']))]=0
df['above'][df['above']<(np.mean(df['Close'])+ 1.5*np.std(df['Close']))]=0


derbyA=df['above']
dfTestA=df['above']
H=derbyA[(derbyA>0)]

derbyB=df['below']
dfTestB=df['below']
L=derbyB[(derbyB>0)]
## prints are for debugging
#print(df['below'])
#sys.exit()
#print(L)
#print(H)

##this is the original graph 

import matplotlib.pyplot as plt
import pandas as pd
plt.clf()
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()
#sns.set(font_scale=1.5)
###df=pd.DataFrame()
rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='o', color='navy', scatter_kws={'s':100})
#rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='*', color='yellow', scatter_kws={'s':50})

zp = sns.regplot(x=df.index, y='Close', data=df, ci=None, marker='*', color='white', scatter_kws={'s':50})

#####usingi numpy works 
#####rp = sns.regplot(x=X1, y=Y1, ci=None, color='green')

y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
z_rp = rp.get_lines()[0].get_ydata()
###print(y_rp)
###print("that was y_rp")
print("THIS GUY IS WHAT ")
print(z_rp[-5:])
sns.regplot(x=x_rp, y=z_rp + ( 2* np.std(z_rp)), color='g')
print("STDDEV:")
mystd=df['Close'].std()
print("MEAN:")
mymean=df['Close'].mean()
print(" std: ",mystd," mean:",mymean)
print(df['Close'][df['Close']>df['Close'].std()].count())
sns.lineplot(x=x_rp, y=y_rp + ( .25* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (.25* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( .5* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (.5* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( .75* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (.75* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( 3* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (3* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( 1.25* np.std(y_rp)), color='g')
sns.lineplot(x=x_rp, y=y_rp - (1.25* np.std(y_rp)), color='purple')
sns.lineplot(x=x_rp, y=y_rp + ( 1.5* np.std(y_rp)), color='g')
sns.lineplot(x=x_rp, y=y_rp - (1.5* np.std(y_rp)), color='purple')
sns.lineplot(x=x_rp, y=y_rp + ( 1.75* np.std(y_rp)), color='g')
sns.lineplot(x=x_rp, y=y_rp - (1.75* np.std(y_rp)), color='purple')
sns.lineplot(x=x_rp, y=y_rp + ( 4* np.std(y_rp)), color='g')
sns.lineplot(x=x_rp, y=y_rp - (4* np.std(y_rp)), color='purple')
plt.legend(title='TESTING ', loc='center left', labels=['YES','NO'],bbox_to_anchor=(.5, -.2),
          fancybox=True, shadow=True, ncol=5)
plt.title("LRC: "+stockName.upper()+"   from:"+str(myStart)+"   to:"+str(myEnd))
#plt.show()
#sns.ylabel('Stock Price')
#sns.xlabel('Number of Days')

print(" can i get to them this way ? ")
testa=(y_rp + ( 1.5* np.std(y_rp)))
testb=df['Close']
testc=df['Close']+df['Close'].std()*1.5
testd=df['Close'].std()
print("testa")
print(testa[-5:])
print("testb")
print(testb[-5:])
print(testc[-5:])
print(testd)
print("testc")
print (len(testa))
print(len(testb))
print(" can i do it here ? ")
#print(testa-testb)
#sns.regplot(x=x_rp, y=y_rp + ( 2* np.std(y_rp)), color='blue')
#sns.lineplot(x=x_rp, y=y_rp - (2* np.std(y_rp)), color='r')

'''
sns.lineplot(x=x_rp, y=y_rp + ( 3* np.std(y_rp)), color='yellow')
sns.lineplot(x=x_rp, y=y_rp - (3* np.std(y_rp)), color='yellow')
sns.lineplot(x=x_rp, y=y_rp + ( 4* np.std(y_rp)), color='purple')
sns.lineplot(x=x_rp, y=y_rp - (4* np.std(y_rp)), color='purple')
'''
### trying for colorization ?
#sns.regplot(x=df.index,y=df['Close']<np.mean(df['Close'])-2*np.std(df['Close']),data=df, ci=None, marker='*', color='red', scatter_kws={'s':50})
#sns.lineplot(x=x_rp, y=y_rp - (2* np.std(y_rp)), color='r')
##df['above'][df['above']<(np.mean(df['Close'])+ 1.5*np.std(df['Close']))]=0

##sns.regplot(x=df.index, y=df['below'], ci=None, color='r')
##sns.regplot(x=df.index, y='above', data=df, ci=None, color='g')
##print(x_rp)

###################################################
'''
plt.figure(figsize=(12,10))
###plt.plot(X1,Y2)
plt.ylabel('Stock Price')
plt.xlabel('Number of Days')
plt.legend(['More Testing'])
plt.title('Title testing')
plt.grid()
plt.show()
'''
testing=y_rp + ( 1.5* np.std(y_rp))
print(testing[-5:])
""" maybe I can get to it this way """ 
""" but i also don't want to forget the SMA's approximate the regression channels""" 
""" if you were to check ... SMA for 6 mos, 3 mos, 5 years etc ... then it will show which channel he is """ 
#print("1.5+",y_rp + ( 1.5* np.std(y_rp)))
#print("1.5-",y_rp - ( 1.5* np.std(y_rp)))

#@title : new quadratic regression function : 
#@title Function that defines the quadratic regression 

def myFitter(polyLevel,myFrame,myName):
    """
    This function takes a dataframe and returns
    quadratic fit and plot 
    """
    print( myName + ". was passed")
    new=myFrame.to_numpy()
    x=range(len(myFrame))
    print(x)
    import numpy as np
    import matplotlib.pyplot as plt
    plt.clf()
    # poly model 
    model=np.poly1d(np.polyfit(x,new,polyLevel))
    line=np.linspace(1,len(df),100)
    #secondDerivative=np.polyder(model,2)
    #roots=np.roots(np.polyder(model,2))
    #print("second derivative:")
    #print(secondDerivative)
    #print(roots)
    plt.title(stockName+" from : "+myStart+" to "+myEnd)
    plt.scatter(x,new)
    plt.plot(line,model(line))
    #plt.show()
    print("the model is : " + str(model) + " \nfirst derivative : " + str(np.polyder(model)))
    print("second derivative: "+str(np.polyder(model,2)))
    print(" roots:"+str(np.roots(np.polyder(model,2))))
    from numpy.fft import fft,ifft
    # fft : 
    #X=np.fft(y_rp)# this did not work, had to import the numpy fft 
    X=fft(new)
    N=len(X)
    n=np.arange(N)
    T=N/2
    freq=n/T
    plt.figure(2)
    plt.stem(freq,np.abs(X),'b',markerfmt=" ",basefmt="-b")
    #plt.show()

#@title : call the new quadratic function :
#@title Call the regression function 
print(" this can be used to see when you would optimal buy call or put")
fitLevel=3
myFitter(fitLevel,df['Close'],"Actual values were passed : "+str(fitLevel)+"\n ")
myFitter(fitLevel,df['percentChange'].dropna(),"percentage:"+str(fitLevel)+"\n")
fitLevel=1
print(" this first image is the slope of the stock's price action")
myFitter(fitLevel,df['Close'],"Actual values were passed : "+str(fitLevel)+"\n ")
print(" this is the slope of the percentage change in the stock ")
myFitter(fitLevel,df['percentChange'].dropna(),"percentage:"+str(fitLevel)+"\n")
#sys.exit()

##@title : make linear regression chart : 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
sns.set(font_scale=1.5)
plt.figure(figsize=(12,10))
#print(L.head())
myMin=min(L)
myMax=max(H)
myDiff=(myMin-myMax)/2
#plt.ylim(myMin-10,myMax+10)
rp = sns.regplot(x=df.index, y='Close', data=df, ci=None, color='black')
y_rp = rp.get_lines()[0].get_ydata()
x_rp = rp.get_lines()[0].get_xdata()
print(y_rp)
print("that was y_rp")
sns.lineplot(x=x_rp, y=y_rp + ( .5* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (.5* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( 1* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (1* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( 1.5* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (1.5* np.std(y_rp)), color='r')
sns.lineplot(x=x_rp, y=y_rp + ( 2* np.std(y_rp)), color='b')
sns.lineplot(x=x_rp, y=y_rp - (2* np.std(y_rp)), color='r')

sns.lineplot(x=x_rp, y=y_rp + ( 3* np.std(y_rp)), color='yellow')
sns.lineplot(x=x_rp, y=y_rp - (3* np.std(y_rp)), color='yellow')

sns.lineplot(x=x_rp, y=y_rp + ( 4* np.std(y_rp)), color='purple')
sns.lineplot(x=x_rp, y=y_rp - (4* np.std(y_rp)), color='purple')

sns.regplot(x=df.index, y=df['below'], ci=None, color='r')
sns.regplot(x=df.index, y='above', data=df, ci=None, color='g')

plt.xlabel('')
plt.ylabel('Price')
#plt.show()

print("now what you need is something that shows when it is above or below this \n say you can find that things are outside of the range . scan for that , then take the \n most recent data and push it through that function above for the shark fins")

#@title : do fft using numpy : 
from numpy.fft import fft,ifft
# fft : 
#X=np.fft(y_rp)# this did not work, had to import the numpy fft 
X=fft(y_rp)
N=len(X)
n=np.arange(N)
T=N/2
freq=n/T

plt.figure(2)
plt.stem(freq,np.abs(X),'b',markerfmt=" ",basefmt="-b")
#plt.show()

"""Version 3 included the new ETF reports

"""
