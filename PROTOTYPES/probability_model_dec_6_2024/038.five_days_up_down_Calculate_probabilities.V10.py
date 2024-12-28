import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
import numpy as np
from glob import glob

def download_data(ticker, period):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def calculate_consecutive_trends(df):
    df['Trend'] = np.sign(df['Close'].diff())
    df['Consecutive'] = df['Trend'].groupby((df['Trend'] != df['Trend'].shift()).cumsum()).cumcount() + 1
    df['Consecutive'] *= df['Trend']
    df['Pct_Change'] = df['Close'].pct_change() * 100
    return df

def load_historical_data(ticker):
    data_files = glob(f"data/*/{ticker}_analysis_*.csv")
    if not data_files:
        print(f"No historical data found for {ticker}.")
        return pd.DataFrame()

    data_frames = []
    for file in data_files:
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        data_frames.append(df)

    all_data = pd.concat(data_frames)
    return all_data

def calculate_probabilities(df):
    probabilities = {}
    for streak in range(1, 6):
        streak_data = df[df['Consecutive'].abs() == streak]
        total_streaks = len(streak_data)
        if total_streaks == 0:
            continue
        reversal_count = 0
        continuation_count = 0
        for i in range(len(streak_data) - 1):
            if streak_data['Trend'].iloc[i] != streak_data['Trend'].iloc[i + 1]:
                reversal_count += 1
            else:
                continuation_count += 1
        probabilities[streak] = {
            'Reversal': reversal_count / total_streaks,
            'Continuation': continuation_count / total_streaks
        }
    return probabilities

def calculate_impact_of_pct_change(df):
    impact_data = {}
    for streak in range(1, 6):
        streak_data = df[df['Consecutive'].abs() == streak]
        if streak_data.empty:
            continue
        avg_pct_change = streak_data['Pct_Change'].abs().mean()
        impact_data[streak] = avg_pct_change
    return impact_data

def plot_data_with_subplots(df, title, ticker, probabilities, impact_data):
    row_heights = [.1, .5, 1, 1]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                         subplot_titles=(title, 'Market Data', 'Consecutive Days Up/Down', 'Daily Percentage Change'),
                         row_heights=row_heights)

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Data'), row=1, col=1)
    positive_consecutive = df['Consecutive'].apply(lambda x: x if x > 0 else 0)
    negative_consecutive = df['Consecutive'].apply(lambda x: x if x < 0 else 0)
    fig.add_trace(go.Bar(x=df.index, y=positive_consecutive, marker_color='green', name='Consecutive Up'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=negative_consecutive, marker_color='red', name='Consecutive Down'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Pct_Change'], marker_color=df['Pct_Change'].apply(lambda x: 'blue' if x > 0 else 'orange'), name='Percentage Change'), row=4, col=1)

    fig.update_layout(height=1800, showlegend=False)

    image_dir = f"images/{ticker}"
    os.makedirs(image_dir, exist_ok=True)
    image_file = os.path.join(image_dir, f'{ticker}_consecutive_days_and_daily_moves.png')
    fig.write_image(image_file, width=1920, height=1080, scale=2)
    print(f"Plot saved as {image_file}")

    pngs_dir = "PNGS"
    os.makedirs(pngs_dir, exist_ok=True)
    pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_consecutive_days_and_daily_moves.png')
    fig.write_image(pngs_file, width=1920, height=1080, scale=2)
    print(f"Plot saved as {pngs_file}")

    fig.show()

    print("Probabilities of Reversal/Continuation:")
    label_mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    for streak in sorted(probabilities.keys()):
        prob = probabilities[streak]
        print(f"Streak Length: {label_mapping[streak]} days - Reversal: {prob['Reversal']:.2f}, Continuation: {prob['Continuation']:.2f}")

    print("Impact of Percentage Change on Streak Length:")
    for streak in sorted(impact_data.keys()):
        impact = impact_data[streak]
        print(f"Streak Length: {label_mapping[streak]} days - Avg. Percentage Change: {impact:.2f}%")

# Example usage:
if len(sys.argv) < 2:
    print("Usage: python script.py <ticker_symbol> [period]")
    sys.exit(1) 

ticker = sys.argv[1]
period = sys.argv[2] if len(sys.argv) >= 3 else "max"
print("No period specified, using 'max' as the default.")

ticker_data = download_data(ticker, period)
ticker_data = calculate_consecutive_trends(ticker_data)
ticker_data.to_csv(f"data/{datetime.today().strftime('%Y-%m-%d')}/{ticker}_analysis_{datetime.today().strftime('%Y-%m-%d')}.csv")

historical_data = load_historical_data(ticker)
if historical_data.empty:
    print(f"No historical data to process for {ticker}.")
else:
    historical_data = calculate_consecutive_trends(historical_data)

    probabilities = calculate_probabilities(historical_data)
    impact_data = calculate_impact_of_pct_change(historical_data)

    plot_data_with_subplots(ticker_data, title=f"{ticker} Stock", ticker=ticker, probabilities=probabilities, impact_data=impact_data)

