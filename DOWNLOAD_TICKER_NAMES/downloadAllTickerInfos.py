#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from datetime import datetime
import yfinance as yf

def save_stock_data(ticker):
    # Define folder path
    current_date = datetime.today().strftime('%Y_%m_%d')
    folder_path = f'DATA/{current_date}'
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Define stock object
    stock = yf.Ticker(ticker)
    
    # Save stock data
    stock_data = stock.history(period='max')
    stock_data.to_csv(f'{folder_path}/{ticker}_stock_data.csv')
    
    # Save other data types
    data_types = ['actions', 'dividends', 'splits']
    for data_type in data_types:
        data = getattr(stock, data_type)
        try:
            data.to_csv(f'{folder_path}/{ticker}_{data_type}.csv')
        except AttributeError:
            print(f"No {data_type} data available for {ticker}.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a stock ticker as an argument.")
        sys.exit(1)
    
    #ticker = sys.argv[1]
    #ticker = 'AAPL'
    #save_stock_data(ticker)


# In[2]:


#nvda = yf.Ticker("NVDA")


# In[ ]:


#print(dir(nvda))


# In[3]:


import os
import sys
from datetime import datetime
import yfinance as yf
import pandas as pd

def save_stock_data(ticker):
    # Define folder path
    current_date = datetime.today().strftime('%Y_%m_%d')
    folder_path = os.path.join('DATA', ticker, current_date)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    '''
    ##########
    # Define folder path
    current_date = datetime.today().strftime('%Y_%m_%d')
    folder_path = f'DATA/{current_date}'
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    '''
        
    # Define stock object
    stock = yf.Ticker(ticker)
    
    # Save stock data
    stock_data = stock.history(period='max')
    stock_data.to_csv(f'{folder_path}/{ticker}_stock_data.csv')
    
    # Save multiple data
    multiple_data = get_multiple_data(ticker)
    for data_type, data in multiple_data.items():
        if isinstance(data, pd.DataFrame):
            data.to_csv(f'{folder_path}/{ticker}_{data_type}.csv')
        else:
            with open(f'{folder_path}/{ticker}_{data_type}.txt', 'w') as file:
                file.write(str(data))
    
    # Save options data
    options_dates = stock.options
    for date in options_dates:
        options_data = stock.option_chain(date)
        options_data.calls.to_csv(f'{folder_path}/{ticker}_options_calls_{date}.csv')
        options_data.puts.to_csv(f'{folder_path}/{ticker}_options_puts_{date}.csv')

def get_multiple_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    data = {
        #'basic_info': ticker.info,
        'balance_sheet': ticker.balance_sheet,
        'cash_flow': ticker.cashflow,
        'dividends': ticker.dividends,
        'financials': ticker.financials,
        'history': ticker.history(period="20y"),
        'institutional_holders': ticker.institutional_holders,
        'major_holders': ticker.major_holders,
        'mutualfund_holders': ticker.mutualfund_holders,
        'quarterly_balance_sheet': ticker.quarterly_balance_sheet,
        'quarterly_cash_flow': ticker.quarterly_cash_flow,
        'quarterly_financials': ticker.quarterly_financials,
        'splits': ticker.splits,
    }
    return data

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide a stock ticker as an argument.")
        sys.exit(1)
    
    #ticker = sys.argv[1]
    ticker = 'dis'
    save_stock_data(ticker)


# In[ ]:




