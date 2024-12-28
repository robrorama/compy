import os
import sys
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def download_data(ticker, save_path):
    """
    Downloads historical data for the specified ticker using yfinance and saves it as a CSV file.

    Args:
        ticker (str): The ticker symbol to download data for.
        save_path (str): The file path to save the downloaded CSV data.

    Returns:
        pd.DataFrame: The downloaded historical data.
    """
    try:
        print(f"Downloading data for ticker '{ticker}' from yfinance...")
        data = yf.download(ticker, start="2000-01-01", progress=False)
        if data.empty:
            print(f"Error: No data found for ticker '{ticker}'. Please check the ticker symbol.")
            sys.exit(1)
        data.to_csv(save_path)
        print(f"Data for '{ticker}' saved to '{save_path}'.")
        return data
    except Exception as e:
        print(f"Error downloading data for ticker '{ticker}': {e}")
        sys.exit(1)

def load_data(ticker, data_dir):
    """
    Loads historical data for the specified ticker. Downloads the data if not found locally.

    Args:
        ticker (str): The ticker symbol to load data for.
        data_dir (str): The base directory where data is stored.

    Returns:
        pd.DataFrame: The historical data for the ticker.
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    ticker_dir = os.path.join(data_dir, today_str)
    os.makedirs(ticker_dir, exist_ok=True)
    file_path = os.path.join(ticker_dir, f"{ticker}.csv")

    if os.path.isfile(file_path):
        print(f"Loading existing data for '{ticker}' from '{file_path}'.")
        try:
            data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            return data
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
            sys.exit(1)
    else:
        # Download data and save
        data = download_data(ticker, file_path)
        return data

def preprocess_data(df):
    """
    Preprocesses the DataFrame by computing necessary columns.

    Args:
        df (pd.DataFrame): The raw historical data.

    Returns:
        pd.DataFrame: The preprocessed data with additional columns.
    """
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()

    # Compute daily percentage change
    df['Pct_Change'] = df['Adj Close'].pct_change()

    # Determine Gap_Direction
    # 1 for Gap Up, -1 for Gap Down, 0 for no gap
    df['Gap_Direction'] = 0
    df['Gap_Direction'] = np.where(df['Open'] > df['Close'].shift(1), 1, 
                                   np.where(df['Open'] < df['Close'].shift(1), -1, 0))

    # Drop rows with NaN in 'Pct_Change' or 'Gap_Direction'
    df.dropna(subset=['Pct_Change', 'Gap_Direction'], inplace=True)

    return df

def probability_after_gap(df, days_ahead=5):
    """
    Computes the probabilities of price movements after a gap event.

    Args:
        df (pd.DataFrame): The preprocessed historical data.
        days_ahead (int): Number of days ahead to analyze.

    Returns:
        dict: Dictionary containing probability metrics.
    """
    # Initialize counters
    gap_up_higher = 0
    gap_up_lower = 0
    gap_down_higher = 0
    gap_down_lower = 0
    total_gap_up = 0
    total_gap_down = 0

    # Iterate through the DataFrame
    for idx in range(len(df) - days_ahead):
        current_gap = df['Gap_Direction'].iloc[idx]
        future_return = df['Pct_Change'].iloc[idx + days_ahead]

        if current_gap == 1:
            total_gap_up += 1
            if future_return > 0:
                gap_up_higher += 1
            elif future_return < 0:
                gap_up_lower += 1
        elif current_gap == -1:
            total_gap_down += 1
            if future_return > 0:
                gap_down_higher += 1
            elif future_return < 0:
                gap_down_lower += 1

    # Compute probabilities
    gap_up_higher_prob = gap_up_higher / total_gap_up if total_gap_up > 0 else 0
    gap_up_lower_prob = gap_up_lower / total_gap_up if total_gap_up > 0 else 0
    gap_down_higher_prob = gap_down_higher / total_gap_down if total_gap_down > 0 else 0
    gap_down_lower_prob = gap_down_lower / total_gap_down if total_gap_down > 0 else 0

    return {
        'gap_up_higher_prob': gap_up_higher_prob,
        'gap_up_lower_prob': gap_up_lower_prob,
        'gap_down_higher_prob': gap_down_higher_prob,
        'gap_down_lower_prob': gap_down_lower_prob
    }

def visualize_ticker(ticker, df, days_ahead=5):
    """
    Visualizes various statistics for a given ticker using Plotly.

    Args:
        ticker (str): Ticker symbol (e.g., 'SPY').
        df (pd.DataFrame): Preprocessed historical data for the ticker.
        days_ahead (int): Number of days ahead to analyze in probability_after_gap.
    """
    # 1. Plot Adjusted Close Price and Highlight Gap Days
    fig1 = go.Figure()

    # Add Adjusted Close Price line
    fig1.add_trace(go.Scatter(
        x=df.index,
        y=df['Adj Close'],
        mode='lines',
        name='Adj Close',
        line=dict(color='blue')
    ))

    # Highlight Gap Up days
    gap_up = df[df['Gap_Direction'] == 1]
    fig1.add_trace(go.Scatter(
        x=gap_up.index,
        y=gap_up['Adj Close'],
        mode='markers',
        name='Gap Up',
        marker=dict(color='green', symbol='triangle-up', size=10),
        hovertemplate='Gap Up<br>Date: %{x}<br>Adj Close: %{y:.2f}<extra></extra>'
    ))

    # Highlight Gap Down days
    gap_down = df[df['Gap_Direction'] == -1]
    fig1.add_trace(go.Scatter(
        x=gap_down.index,
        y=gap_down['Adj Close'],
        mode='markers',
        name='Gap Down',
        marker=dict(color='red', symbol='triangle-down', size=10),
        hovertemplate='Gap Down<br>Date: %{x}<br>Adj Close: %{y:.2f}<extra></extra>'
    ))

    # Update layout
    fig1.update_layout(
        title=f'{ticker}: Adjusted Close Price with Gap Events Highlighted',
        xaxis_title='Date',
        yaxis_title='Adjusted Close Price ($)',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=0, y=1.0)
    )

    fig1.show()

    # 2. Distribution of Daily Returns (Pct_Change)
    # Create a histogram with KDE-like curve using Plotly
    fig2 = px.histogram(
        df['Pct_Change'],
        nbins=50,
        title=f'{ticker}: Distribution of Daily % Changes',
        labels={'value': 'Daily % Change'},
        opacity=0.75,
        marginal="box",  # Adds a box plot on top
        hover_data=df['Pct_Change']
    )

    fig2.update_traces(marker_color='purple')

    fig2.update_layout(
        xaxis_title='Daily % Change',
        yaxis_title='Frequency',
        template='plotly_white'
    )

    fig2.show()

    # 3. Comparing Probabilities as a Bar Chart
    results = probability_after_gap(df, days_ahead=days_ahead)

    # Create a small dataframe for plotting these probabilities
    prob_df = pd.DataFrame({
        'Condition': ['Gap Up Higher', 'Gap Up Lower', 'Gap Down Higher', 'Gap Down Lower'],
        'Probability': [
            results['gap_up_higher_prob'], 
            results['gap_up_lower_prob'], 
            results['gap_down_higher_prob'], 
            results['gap_down_lower_prob']
        ]
    })

    fig3 = px.bar(
        prob_df,
        x='Condition',
        y='Probability',
        title=f'{ticker}: Probability {days_ahead} Days After Gap',
        labels={'Probability': 'Probability'},
        text=prob_df['Probability'].apply(lambda x: f"{x:.2f}"),
        color='Condition',
        color_discrete_sequence=px.colors.sequential.Coolwarm
    )

    fig3.update_traces(textposition='outside')
    fig3.update_layout(
        yaxis=dict(range=[0, 1]),
        xaxis_title='Condition',
        yaxis_title='Probability',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        template='plotly_white'
    )

    fig3.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))

    fig3.show()

def main():
    """
    Main function to execute the visualization script.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualize stock data with gap events using yfinance data and Plotly.")
    parser.add_argument('ticker', type=str, help='Ticker symbol (e.g., SPY)')
    parser.add_argument('--data_dir', type=str, default='data', help='Base directory to store/download data')
    parser.add_argument('--days_ahead', type=int, default=5, help='Number of days ahead to analyze after a gap event')

    args = parser.parse_args()

    ticker = args.ticker.upper()
    data_dir = args.data_dir
    days_ahead = args.days_ahead

    # Load or download data
    raw_data = load_data(ticker, data_dir)

    # Preprocess data
    df = preprocess_data(raw_data)

    if df.empty:
        print(f"Error: No valid data available for ticker '{ticker}' after preprocessing.")
        sys.exit(1)

    # Visualize data
    visualize_ticker(ticker, df, days_ahead=days_ahead)

if __name__ == "__main__":
    main()
