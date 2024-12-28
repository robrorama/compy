import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from yahoo_earnings_calendar import YahooEarningsCalendar
import os

plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 300

stockName = ""

# Function to get today's date-stamped data folder
def get_data_folder():
    today = date.today().strftime("%Y-%m-%d")
    data_folder = f"data/{today}"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    return data_folder

# Function to save data locally
def save_data(ticker, data):
    data_folder = get_data_folder()
    ticker_folder = os.path.join(data_folder, ticker)
    if not os.path.exists(ticker_folder):
        os.makedirs(ticker_folder)
    file_path = os.path.join(ticker_folder, "data.csv")
    data.to_csv(file_path)
    return file_path

# Function to load data if it exists locally
def load_data(ticker):
    data_folder = get_data_folder()
    ticker_folder = os.path.join(data_folder, ticker)
    file_path = os.path.join(ticker_folder, "data.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    return None

# Function to fetch data (from cache or download)
def fetch_data(ticker, period="1y"):
    data = load_data(ticker)
    if data is not None:
        st.write(f"Loaded cached data for {ticker}.")
        return data
    else:
        st.write(f"Downloading data for {ticker}...")
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, start=f"{date.today().year}-01-01", end=f"{date.today().year}-12-31")
        save_data(ticker, data)
        return data

# Stock selection
def select_stock():
    global stockName
    stockName = st.sidebar.text_input("Enter the stock symbol", key="stock_input")
    if stockName:
        st.sidebar.title(f"Selected Stock: {stockName}")
    return stockName

# Stock information display
def stock_info():
    if not stockName:
        st.warning("Please select a stock first.")
        return

    stock = yf.Ticker(stockName)
    try:
        st.write("Fetching stock information...")
        data = fetch_data(stockName)
        st.write("Recent Data:", data.tail())
    except Exception as e:
        st.error(f"Error fetching stock info: {e}")

# MACD report
def macd_report():
    if not stockName:
        st.warning("Please select a stock first.")
        return

    try:
        df = fetch_data(stockName)
        df['12_EMA'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['26_EMA'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['12_EMA'] - df['26_EMA']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        st.write("MACD Analysis")
        st.line_chart(df[['MACD', 'Signal']])

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df['MACD'], label="MACD", color="blue")
        ax.plot(df['Signal'], label="Signal", color="red")
        ax.set_title(f"MACD Report for {stockName}")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating MACD report: {e}")

# Sector comparison
def sector_comparison():
    tickers = ['XLP', 'XLY', 'XLK', 'XLV', 'XLU', 'XLB', 'XLRE']
    dataframes = {ticker: fetch_data(ticker) for ticker in tickers}
    closing_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes.items()})
    normalized = closing_prices / closing_prices.iloc[0] * 100
    st.line_chart(normalized)

# Index comparison
def index_comparison():
    tickers = ['IWM', 'DIA', 'QQQ', 'SPY']
    dataframes = {ticker: fetch_data(ticker) for ticker in tickers}
    closing_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes.items()})
    normalized = closing_prices / closing_prices.iloc[0] * 100
    st.line_chart(normalized)

# Gold vs Dollar comparison
def gold_vs_dollar():
    tickers = ['GOLD', 'UUP', 'UDN']
    dataframes = {ticker: fetch_data(ticker) for ticker in tickers}
    closing_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes.items()})
    normalized = closing_prices / closing_prices.iloc[0] * 100
    fig, ax = plt.subplots()
    normalized.plot(ax=ax)
    ax.set_title("Gold vs Dollar")
    st.pyplot(fig)

# Currency comparison
def currency_comparison():
    tickers = ['UUP', 'UDN', 'FXE', 'EUFX', 'FXB', 'CYB']
    dataframes = {ticker: fetch_data(ticker) for ticker in tickers}
    closing_prices = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes.items()})
    normalized = closing_prices / closing_prices.iloc[0] * 100
    fig, ax = plt.subplots()
    normalized.plot(ax=ax)
    ax.set_title("Currency Comparison")
    ax.set_ylim(85, 110)
    st.pyplot(fig)

# Stock links
def stock_links():
    if not stockName:
        st.warning("Please select a stock first.")
        return

    st.subheader(f"Quick Links for {stockName}")
    st.markdown(f"[Yahoo Finance](https://finance.yahoo.com/quote/{stockName})")
    st.markdown(f"[Google Finance](https://www.google.com/finance/quote/{stockName})")
    st.markdown(f"[Nasdaq Earnings](https://www.nasdaq.com/market-activity/stocks/{stockName}/earnings)")
    st.markdown(f"[MarketBeat Earnings](https://www.marketbeat.com/stocks/NASDAQ/{stockName}/earnings/)")
    st.markdown(f"[Barchart Overview](https://www.barchart.com/stocks/quotes/{stockName}/overview)")

# Earnings calendar
def earnings_calendar():
    st.write("Earnings Calendar")
    try:
        calendar = YahooEarningsCalendar()
        start = datetime.now().date()
        end = start + timedelta(days=180)  # Extend range to six months
        try:
            earnings = pd.DataFrame(calendar.earnings_between(start, end))
            if not earnings.empty:
                st.write(earnings)
            else:
                st.write("No earnings found in the specified range.")
        except IndexError:
            st.error("Error fetching earnings data: No earnings found in the specified range.")
    except Exception as e:
        st.error(f"Error fetching earnings data: {e}")

# Re-render all plots with percentage changes
def rerender_percentage_changes():
    # Sector comparison with percentage changes
    tickers_sectors = ['XLP', 'XLY', 'XLK', 'XLV', 'XLU', 'XLB', 'XLRE']
    dataframes_sectors = {ticker: fetch_data(ticker) for ticker in tickers_sectors}
    closing_prices_sectors = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes_sectors.items()})
    percentage_changes_sectors = closing_prices_sectors.pct_change() * 100
    st.subheader("Sector Comparison (Percentage Changes)")
    st.line_chart(percentage_changes_sectors)

    # Index comparison with percentage changes
    tickers_indexes = ['IWM', 'DIA', 'QQQ', 'SPY']
    dataframes_indexes = {ticker: fetch_data(ticker) for ticker in tickers_indexes}
    closing_prices_indexes = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes_indexes.items()})
    percentage_changes_indexes = closing_prices_indexes.pct_change() * 100
    st.subheader("Index Comparison (Percentage Changes)")
    st.line_chart(percentage_changes_indexes)

    # Gold vs Dollar comparison with percentage changes
    tickers_gold = ['GOLD', 'UUP', 'UDN']
    dataframes_gold = {ticker: fetch_data(ticker) for ticker in tickers_gold}
    closing_prices_gold = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes_gold.items()})
    percentage_changes_gold = closing_prices_gold.pct_change() * 100
    st.subheader("Gold vs Dollar (Percentage Changes)")
    st.line_chart(percentage_changes_gold)

    # Currency comparison with percentage changes
    tickers_currency = ['UUP', 'UDN', 'FXE', 'EUFX', 'FXB', 'CYB']
    dataframes_currency = {ticker: fetch_data(ticker) for ticker in tickers_currency}
    closing_prices_currency = pd.DataFrame({ticker: df['Close'] for ticker, df in dataframes_currency.items()})
    percentage_changes_currency = closing_prices_currency.pct_change() * 100
    st.subheader("Currency Comparison (Percentage Changes)")
    st.line_chart(percentage_changes_currency)

# Streamlit UI
st.sidebar.header("Stock Analysis Tool")
select_stock()

# Use radio buttons for a single selection at a time
options = st.sidebar.radio(
    "Select Analysis",
    (
        "Show Stock Info",
        "Show MACD Report",
        "Compare Sectors",
        "Compare Indexes",
        "Gold vs Dollar",
        "Compare Currencies",
        "Show Earnings Calendar",
        "Show Stock Links",
        "Re-render All Plots as Percentage Changes"
    ),
    key="analysis_options"
)

# Display the selected analysis
def handle_selection(option):
    if option == "Show Stock Info":
        stock_info()
    elif option == "Show MACD Report":
        macd_report()
    elif option == "Compare Sectors":
        sector_comparison()
    elif option == "Compare Indexes":
        index_comparison()
    elif option == "Gold vs Dollar":
        gold_vs_dollar()
    elif option == "Compare Currencies":
        currency_comparison()
    elif option == "Show Earnings Calendar":
        earnings_calendar()
    elif option == "Show Stock Links":
        stock_links()
    elif option == "Re-render All Plots as Percentage Changes":
        rerender_percentage_changes()

handle_selection(options)
