import sys
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

def calculate_macd(dataframe, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD, signal, and histogram from the provided dataframe.
    """
    exponential_moving_average_fast = dataframe['Close'].ewm(span=fast_period, adjust=False).mean()
    exponential_moving_average_slow = dataframe['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exponential_moving_average_fast - exponential_moving_average_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(dataframe, period=14):
    """
    Calculate Relative Strength Index (RSI) from the provided dataframe.
    """
    delta = dataframe['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi

def find_peaks_troughs(dataframe, order=5):
    """
    Find local maxima and minima using scipy.signal.argrelextrema.

    Args:
        dataframe: Pandas DataFrame with 'Close' prices.
        order: How many points on each side to use for the comparison.

    Returns:
        peaks: List of (index, price) tuples for peaks.
        troughs: List of (index, price) tuples for troughs.
    """
    # Get indices of local maxima and minima
    maxima_indices = argrelextrema(dataframe['Close'].values, np.greater, order=order)[0]
    minima_indices = argrelextrema(dataframe['Close'].values, np.less, order=order)[0]

    peaks = [(i, dataframe['Close'].values[i]) for i in maxima_indices]
    troughs = [(i, dataframe['Close'].values[i]) for i in minima_indices]

    return peaks, troughs

def plot_data(dataframe, ticker, peaks, troughs):
    """
    Plot candlestick chart, volume, MACD, RSI, peak-trough I-bars, and recovery boxes.
    """
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03,
                           row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candlestick
    figure.add_trace(go.Candlestick(
        x=dataframe.index,
        open=dataframe['Open'],
        high=dataframe['High'],
        low=dataframe['Low'],
        close=dataframe['Close'],
        name='Candlestick'
    ), row=1, col=1)

    # Peaks and Troughs (markers)
    peak_dates = [dataframe.index[i] for (i, _) in peaks]
    peak_prices = [price for (_, price) in peaks]
    trough_dates = [dataframe.index[i] for (i, _) in troughs]
    trough_prices = [price for (_, price) in troughs]

    figure.add_trace(go.Scatter(
        x=peak_dates,
        y=peak_prices,
        mode='markers',
        name='Peaks',
        marker=dict(color='lime', size=10)
    ), row=1, col=1)

    figure.add_trace(go.Scatter(
        x=trough_dates,
        y=trough_prices,
        mode='markers',
        name='Troughs',
        marker=dict(color='red', size=10)
    ), row=1, col=1)

    # I-bar drawdowns
    peak_indices = [p[0] for p in peaks]
    peak_values = [p[1] for p in peaks]
    trough_indices = [t[0] for t in troughs]
    trough_values = [t[1] for t in troughs]

    pairs = []
    i = 0
    j = 0
    while i < len(peaks) and j < len(troughs):
        if peak_indices[i] < trough_indices[j]:
            pairs.append((peak_indices[i], peak_values[i],
                          trough_indices[j], trough_values[j]))
            i += 1
        else:
            j += 1
    while i < len(peaks) - 1:
        pairs.append((peak_indices[i], peak_values[i],
                      trough_indices[j - 1], trough_values[j - 1]))
        i += 1

    for (peak_index, peak_price, trough_index, trough_price) in pairs:
        peak_x = dataframe.index[peak_index]
        trough_x = dataframe.index[trough_index]
        # Top horizontal
        figure.add_shape(
            type="line",
            x0=peak_x,
            y0=peak_price,
            x1=trough_x,
            y1=peak_price,
            line=dict(color="RoyalBlue", width=2),
            row=1,
            col=1
        )
        # Bottom horizontal
        figure.add_shape(
            type="line",
            x0=peak_x,
            y0=trough_price,
            x1=trough_x,
            y1=trough_price,
            line=dict(color="RoyalBlue", width=2),
            row=1,
            col=1
        )
        # Vertical connector
        midpoint_x = peak_x + (trough_x - peak_x) / 2
        figure.add_shape(
            type="line",
            x0=midpoint_x,
            y0=peak_price,
            x1=midpoint_x,
            y1=trough_price,
            line=dict(color="RoyalBlue", width=2),
            row=1,
            col=1
        )

    # Yellow Recovery Boxes
    for i in range(len(pairs) - 1):
        _, _, trough_index, trough_price = pairs[i]
        next_peak_index, next_peak_price, _, _ = pairs[i+1]

        trough_x = dataframe.index[trough_index]
        next_peak_x = dataframe.index[next_peak_index]

        figure.add_shape(
            type="rect",
            x0=trough_x,
            y0=trough_price,
            x1=next_peak_x,
            y1=next_peak_price,
            line=dict(color="gold", width=0),
            fillcolor="gold",
            opacity=0.3,
            row=1,
            col=1
        )


    # Volume
    figure.add_trace(go.Bar(
        x=dataframe.index,
        y=dataframe['Volume'],
        name='Volume',
        marker_color='rgba(158,202,225,0.5)'
    ), row=2, col=1)

    # MACD
    figure.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe['MACD'],
        name='MACD',
        line=dict(color='blue')
    ), row=3, col=1)

    figure.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe['Signal'],
        name='Signal',
        line=dict(color='orange')
    ), row=3, col=1)

    figure.add_trace(go.Bar(
        x=dataframe.index,
        y=dataframe['Histogram'],
        name='Histogram',
        marker_color='grey'
    ), row=3, col=1)

    # RSI
    figure.add_trace(go.Scatter(
        x=dataframe.index,
        y=dataframe['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=4, col=1)

    figure.add_hline(
        y=70,
        line_dash="dot",
        annotation_text="Overbought",
        annotation_position="bottom right",
        row=4,
        col=1
    )
    figure.add_hline(
        y=30,
        line_dash="dot",
        annotation_text="Oversold",
        annotation_position="top right",
        row=4,
        col=1
    )

    figure.update_layout(
        title=f'{ticker} Stock Analysis (Last 1 Year)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1000
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    figure.update_yaxes(title_text="MACD", row=3, col=1)
    figure.update_yaxes(title_text="RSI", row=4, col=1)

    figure.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <ticker> <order>")
        sys.exit(1)

    ticker = sys.argv[1]
    order = int(sys.argv[2])

    # Download one year of data
    dataframe = yf.download(ticker, period='1y')

    # Calculate MACD
    dataframe['MACD'], dataframe['Signal'], dataframe['Histogram'] = calculate_macd(dataframe)

    # Calculate RSI
    dataframe['RSI'] = calculate_rsi(dataframe)

    # Find peaks and troughs
    peaks, troughs = find_peaks_troughs(dataframe, order=order)

    # Plot the results
    plot_data(dataframe, ticker, peaks, troughs)
