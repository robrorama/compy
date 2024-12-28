import sys
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_peaks_troughs(data, threshold=0.04, col='Close'):
    """
    Alternate between searching for a local peak and a local trough based on data[col].
    Only register a new peak/trough if price difference from the last extremum is above the threshold.
    Returns lists: peaks, troughs
    """
    peaks = []
    troughs = []
    searching_for_peak = True
    last_extremum_price = None
    prices = data[col].values

    for i in range(1, len(prices) - 1):
        if searching_for_peak:
            # Potential peak
            if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
                if (last_extremum_price is None or
                   (prices[i] - last_extremum_price) / last_extremum_price > threshold):
                    peaks.append((i, prices[i]))
                    last_extremum_price = prices[i]
                    searching_for_peak = False
        else:
            # Potential trough
            if prices[i] < prices[i - 1] and prices[i] < prices[i + 1]:
                if (last_extremum_price is None or
                   (last_extremum_price - prices[i]) / last_extremum_price > threshold):
                    troughs.append((i, prices[i]))
                    last_extremum_price = prices[i]
                    searching_for_peak = True

    return peaks, troughs

def plot_data(data, ticker, peaks, troughs):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlestick'),
        row=1, col=1
    )

    # Markers for peaks and troughs
    peak_dates = [data.index[i] for (i, _) in peaks]
    peak_prices = [p for (_, p) in peaks]
    trough_dates = [data.index[i] for (i, _) in troughs]
    trough_prices = [t for (_, t) in troughs]

    fig.add_trace(go.Scatter(
        x=peak_dates,
        y=peak_prices,
        mode='markers',
        name='Peaks',
        marker=dict(color='lime', size=10)),
        row=1, col=1
    )

    fig.add_trace(go.Scatter(
        x=trough_dates,
        y=trough_prices,
        mode='markers',
        name='Troughs',
        marker=dict(color='red', size=10)),
        row=1, col=1
    )

    # I-bar lines
    p_idx = [p[0] for p in peaks]
    p_val = [p[1] for p in peaks]
    t_idx = [t[0] for t in troughs]
    t_val = [t[1] for t in troughs]

    pairs = []
    i = 0
    j = 0
    while i < len(peaks) and j < len(troughs):
        if p_idx[i] < t_idx[j]:
            pairs.append((p_idx[i], p_val[i], t_idx[j], t_val[j]))
            i += 1
            j += 1
        else:
            j += 1

    for (peak_i, peak_price, trough_i, trough_price) in pairs:
        peak_x = data.index[peak_i]
        trough_x = data.index[trough_i]

        fig.add_shape(
            type="line",
            x0=peak_x, y0=peak_price,
            x1=trough_x, y1=peak_price,
            line=dict(color="RoyalBlue", width=2),
            row=1, col=1
        )
        fig.add_shape(
            type="line",
            x0=peak_x, y0=trough_price,
            x1=trough_x, y1=trough_price,
            line=dict(color="RoyalBlue", width=2),
            row=1, col=1
        )
        midpoint_x = peak_x + (trough_x - peak_x) / 2
        fig.add_shape(
            type="line",
            x0=midpoint_x, y0=peak_price,
            x1=midpoint_x, y1=trough_price,
            line=dict(color="RoyalBlue", width=2),
            row=1, col=1
        )

    # Volume
    fig.add_trace(go.Bar(
        x=data.index, 
        y=data['Volume'], 
        name='Volume', 
        marker_color='rgba(158,202,225,0.5)'),
        row=2, col=1
    )

    # MACD
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['MACD'], 
        name='MACD', 
        line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Signal'], 
        name='Signal', 
        line=dict(color='orange')),
        row=3, col=1
    )
    fig.add_trace(go.Bar(
        x=data.index, 
        y=data['Histogram'], 
        name='Histogram', 
        marker_color='grey'),
        row=3, col=1
    )

    # RSI
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['RSI'], 
        name='RSI', 
        line=dict(color='purple')),
        row=4, col=1
    )
    fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought",
                  annotation_position="bottom right", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold",
                  annotation_position="top right", row=4, col=1)

    # Layout
    fig.update_layout(
        title=f'{ticker} Stock Analysis (Last 1 Year)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=1000
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <ticker> <threshold> <smooth_days>")
        sys.exit(1)

    ticker = sys.argv[1]
    threshold = float(sys.argv[2])
    smooth_days = int(sys.argv[3])

    # Download data
    data = yf.download(ticker, period='1y')
    if data.empty:
        print("No data downloaded. Check ticker or connection.")
        sys.exit(1)

    # MACD, RSI
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    data['RSI'] = calculate_rsi(data)

    # Compute a rolling mean for smoothing detection
    data['Smoothed_Close'] = data['Close'].rolling(window=smooth_days).mean()
    data.dropna(subset=['Smoothed_Close'], inplace=True)

    # If everything got dropped, just plot the candlesticks
    if data.empty:
        print("No data left after smoothing. Try a smaller smoothing window.")
        sys.exit(1)

    # Get peaks/troughs from smoothed data
    peaks, troughs = find_peaks_troughs(data, threshold=threshold, col='Smoothed_Close')

    # Plot
    plot_data(data, ticker, peaks, troughs)
