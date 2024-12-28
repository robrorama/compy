#!/usr/bin/env python3

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timedelta
import statsmodels.api as sm
from fpdf import FPDF

# Function to download data and save to CSV if it does not exist locally
def download_data(ticker, start, end, period='1d'):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(data_file):
        print(f"Data for {ticker} loaded from disk.")
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        print(f"Downloading data for {ticker}")
        df = yf.download(ticker, start=start, end=end, period=period)
        if df.empty:
            print(f"Failed to download data for {ticker}.")
            sys.exit(1)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")

    return df

# Define functions for plotting and saving high-resolution images
def plot_and_save(df, title, filename):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(16, 8))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title(title)
    plt.legend()
    plt.savefig(f'PNG/{filename}', dpi=300)

# Plot Linear Regression Channels (LRC) with moving averages
def plot_lrc(df, stockName, start, end):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(16, 8))
    sns.lineplot(x='Date', y='Close', data=df, label='Close Price')
    sns.lineplot(x='Date', y='SMA30', data=df, label='SMA30', color='green')
    sns.lineplot(x='Date', y='SMA50', data=df, label='SMA50', color='orange')
    sns.lineplot(x='Date', y='SMA100', data=df, label='SMA100', color='blue')
    sns.lineplot(x='Date', y='SMA200', data=df, label='SMA200', color='purple')
    sns.lineplot(x='Date', y='SMA300', data=df, label='SMA300', color='red')
    plt.title(f"LRC: {stockName.upper()} from {start} to {end}")
    plt.legend()
    plt.savefig(f"PNG/LRC_{stockName}_{start}_{end}.png", dpi=300)

# Plot Q-Q plot and distribution
def plot_qq(df, colName, stockName, start, end):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.histplot(df[colName], kde=False, ax=ax[0], color='blue')
    ax[0].set_title(f'Distribution of {stockName} {colName}', fontsize=16)
    sm.qqplot(df[colName], line='s', ax=ax[1])
    ax[1].set_title(f'Q-Q plot of {stockName} {colName}', fontsize=16)
    plt.savefig(f"PNG/QQ_{stockName}_{colName}_{start}_{end}.png", dpi=300)

# Calculate and plot consecutive up/down days
def plot_consecutive_days(df, stockName, start, end):
    df2 = df['simple_rtn'].copy(deep=True)
    df2[df2 > 0] = 1
    df2[df2 < 0] = -1
    df3 = df2.copy(deep=True)
    tally = 1
    old = 1
    runningSum = 0
    for value in df3:
        if not np.isnan(value):
            combined = int(old) + int(value)
            if combined == 0:
                runningSum = 0
            elif combined < 0:
                runningSum -= 1
            elif combined > 0:
                runningSum += 1
            tally += 1
            old = value
            df3.iloc[tally - 1] = runningSum
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(16, 8))
    sns.lineplot(data=df3, color='green')
    plt.title(f"Consecutive Up/Down Days for {stockName} from {start} to {end}")
    plt.savefig(f"PNG/Consecutive_UpDown_Days_{stockName}_{start}_{end}.png", dpi=300)

# Calculate and plot first and second derivatives
def plot_derivatives(df, stockName, start, end):
    df['first_derivative'] = df['Close'].diff()
    df['second_derivative'] = df['first_derivative'].diff()
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(16, 8))
    sns.lineplot(x='Date', y='first_derivative', data=df, label='First Derivative', color='gold')
    sns.lineplot(x='Date', y='second_derivative', data=df, label='Second Derivative', color='brown')
    plt.title(f"First and Second Derivatives for {stockName} from {start} to {end}")
    plt.legend()
    plt.savefig(f"PNG/Derivatives_{stockName}_{start}_{end}.png", dpi=300)

# Perform quadratic regression and plot
def myFitter(polyLevel, myFrame, myName):
    x = range(len(myFrame))
    model = np.poly1d(np.polyfit(x, myFrame.to_numpy(), polyLevel))
    line = np.linspace(1, len(myFrame), 100)
    plt.figure(figsize=(16, 8))
    plt.scatter(x, myFrame)
    plt.plot(line, model(line))
    plt.title(f"Quadratic Fit for {myName}")
    plt.savefig(f"PNG/Quadratic_Fit_{myName}.png", dpi=300)
    print(f"Model: {model}\nFirst Derivative: {np.polyder(model)}\nSecond Derivative: {np.polyder(model, 2)}")
    fft_analysis(myFrame, myName)

# Perform FFT analysis and plot
def fft_analysis(myFrame, myName):
    X = np.fft.fft(myFrame)
    N = len(X)
    n = np.arange(N)
    T = N / 2
    freq = n / T
    plt.figure(figsize=(16, 8))
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.title(f"FFT Analysis for {myName}")
    plt.savefig(f"PNG/FFT_{myName}.png", dpi=300)

# Identify outliers using the 3-sigma rule
def identify_outliers(row, n_sigmas=3):
    x = row['simple_rtn']
    mu = row['mean']
    sigma = row['std']
    if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
        return 1
    else:
        return 0

# Show outliers chart
def showThreeSD(df, df_outliers, outliers, stockName, start, end):
    fig, ax = plt.subplots()
    ax.plot(df_outliers.index, df_outliers.simple_rtn, color='blue', label='Normal')
    ax.scatter(outliers.index, outliers.simple_rtn, color='red', label='Anomaly')
    ax.set_title(stockName + " (3xstd.dev) stock price moves this period==>" + str(outliers.simple_rtn.count()))
    ax.legend(loc='lower right')
    plt.savefig(f"PNG/3_standard_deviations_graph_{stockName}_{start}_{end}.png", dpi=300)

# Generate regression channel plot with percentage changes and standard deviation lines
def plot_regression_channel(df, stockName, start, end):
    sns.set_style("white")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    myRange = 'SMA30PC'
    df['Date_ordinal'] = df.index.map(datetime.toordinal)
    rp = sns.regplot(x=df['Date_ordinal'], y=myRange, data=df, ci=None, marker='.', color='lawngreen', scatter_kws={'s':75})
    zp = sns.regplot(x=df['Date_ordinal'], y=myRange, data=df, ci=None, marker='.', color='black', scatter_kws={'s':5})

    y_rp = rp.get_lines()[0].get_ydata()
    x_rp = rp.get_lines()[0].get_xdata()

    def printMyStats(myRange):
        mystd = df[myRange].std()
        mymean = df[myRange].mean()
        print(f'StandardDeviation {myRange} PercentageChange:', mystd)
        print(f'Mean {myRange} PercentageChange:', mymean)
        print(df.tail(5))

    printMyStats(myRange)

    def drawLines(deviations, color):
        for i in deviations:
            sns.lineplot(x=x_rp, y=y_rp + (float(i) * np.std(y_rp)), color=color, linewidth=.5, linestyle='-')
            sns.lineplot(x=x_rp, y=y_rp - (float(i) * np.std(y_rp)), color=color, linewidth=.5, linestyle='-')

    deviations_light = ['.25', '.5', '.75', '1', '1.25', '1.5', '1.75']
    deviations_dark = ['1', '2', '3', '4', '5']
    drawLines(deviations_light, 'lightgrey')
    drawLines(deviations_dark, 'black')

    smas = ['SMA30', 'SMA50', 'SMA100', 'SMA200', 'SMA300']
    myColors = ['green', 'orange', 'blue', 'purple', 'red']
    for i in range(len(smas)):
        sns.lineplot(x=df['Date_ordinal'], y=smas[i], data=df, color=myColors[i], linewidth=.6)

    sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
    plt.title(f"LRC: percentage graphs {stockName.upper()} from {start} to {end}")
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.savefig(f'PNG/movingAverages_percentages_{stockName}_{start}_{end}.png', dpi=300)

def convert_images_to_pdf(images_folder, pdf_name):
    pdf_folder = "PDF"
    os.makedirs(pdf_folder, exist_ok=True)
    pdf = FPDF("L", unit="pt", format="legal")
    imagelist = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    for image in imagelist:
        imagename = os.path.join(images_folder, image)
        pdf.add_page()
        pdf.image(imagename, 0, 0, 0, 525)
    pdf_file = os.path.join(pdf_folder, f"{pdf_name}.pdf")
    pdf.output(pdf_file, "F")
    print(f"PDF created: {pdf_file}")

# Main function to run the analysis
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start = '2022-01-01'
    end = datetime.today().strftime('%Y-%m-%d')

    # Ensure PNG folder exists before saving plots
    os.makedirs("PNG", exist_ok=True)

    # Download or load data
    df = download_data(ticker, start, end)

    # Add necessary columns for plotting
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
    df['simple_rtn'] = df['Close'].pct_change()
    df['percentChange'] = df['Close'].pct_change() * 100

    # Generate plots
    plot_lrc(df, ticker, start, end)
    plot_qq(df, 'Close', ticker, start, end)
    plot_consecutive_days(df, ticker, start, end)
    plot_derivatives(df, ticker, start, end)
    plot_regression_channel(df, ticker, start, end)
    myFitter(3, df['Close'], f'{ticker}_Close')
    myFitter(1, df['percentChange'].dropna(), f'{ticker}_percentChange')

    # Process outliers
    df_outliers = df[['simple_rtn']].rolling(window=21).agg(['mean', 'std'])
    df_outliers.columns = df_outliers.columns.droplevel()
    df_outliers = df.join(df_outliers)
    df_outliers['outlier'] = df_outliers.apply(identify_outliers, axis=1)
    outliers = df_outliers.loc[df_outliers['outlier'] == 1, ['simple_rtn']]

    # Show outliers chart
    showThreeSD(df, df_outliers, outliers, ticker, start, end)

    # Convert images to PDF
    convert_images_to_pdf('PNG', f'{ticker}_{start}_{end}')

if __name__ == '__main__':
    main()
