import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import sys

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

def find_peaks_troughs(data, threshold=0.04):
    """
    Alternate between searching for a local peak and then a local trough.
    Only record them if the price swing from the last extremum exceeds 'threshold'.
    """
    peaks = []
    troughs = []

    # We'll track the last recorded extremum price to enforce threshold
    last_extremum_price = None
    searching_for_peak = True

    close_prices = data['Close'].values
    for i in range(1, len(close_prices) - 1):
        # Identify local maxima
        if searching_for_peak:
            if close_prices[i] > close_prices[i-1] and close_prices[i] > close_prices[i+1]:
                # If we don't have a prior extremum, or we exceed threshold from the last extremum
                if last_extremum_price is None or ((close_prices[i] - last_extremum_price) / last_extremum_price) > threshold:
                    peaks.append((i, close_prices[i]))
                    last_extremum_price = close_prices[i]
                    searching_for_peak = False
        else:
            # Identify local minima
            if close_prices[i] < close_prices[i-1] and close_prices[i] < close_prices[i+1]:
                if last_extremum_price is None or ((last_extremum_price - close_prices[i]) / last_extremum_price) > threshold:
                    troughs.append((i, close_prices[i]))
                    last_extremum_price = close_prices[i]
                    searching_for_peak = True

    return peaks, troughs

def plot_data(data, ticker, peaks, troughs):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.2, 0.15, 0.15])

    # -------------------- Candlestick --------------------
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'),
                  row=1, col=1)

    # -------------------- Markers for Peaks & Troughs --------------------
    peak_dates = [data.index[i] for (i, _) in peaks]
    peak_prices = [p for (_, p) in peaks]
    trough_dates = [data.index[i] for (i, _) in troughs]
    trough_prices = [t for (_, t) in troughs]

    fig.add_trace(go.Scatter(x=peak_dates, y=peak_prices, mode='markers',
                             name='Peaks', marker=dict(color='lime', size=10)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=trough_dates, y=trough_prices, mode='markers',
                             name='Troughs', marker=dict(color='red', size=10)),
                  row=1, col=1)

    # -------------------- I-bar Drawdowns --------------------
    # Pair each peak with the next trough in chronological order
    p_idx = [p[0] for p in peaks]
    p_val = [p[1] for p in peaks]
    t_idx = [t[0] for t in troughs]
    t_val = [t[1] for t in troughs]

    pairs = []
    i = j = 0
    while i < len(peaks) and j < len(troughs):
        if p_idx[i] < t_idx[j]:
            # Found a valid peak->trough pair
            pairs.append((p_idx[i], p_val[i], t_idx[j], t_val[j]))
            i += 1
            j += 1
        else:
            # Move to the next trough until we find one after this peak
            j += 1

    # Draw shapes for each pair
    for (peak_i, peak_price, trough_i, trough_price) in pairs:
        peak_x = data.index[peak_i]
        trough_x = data.index[trough_i]

        fig.add_shape(type="line",
                      x0=peak_x, y0=peak_price,
                      x1=trough_x, y1=peak_price,
                      line=dict(color="RoyalBlue", width=2),
                      row=1, col=1)
        fig.add_shape(type="line",
                      x0=peak_x, y0=trough_price,
                      x1=trough_x, y1=trough_price,
                      line=dict(color="RoyalBlue", width=2),
                      row=1, col=1)
        # Midpoint date for vertical line
        midpoint_x = peak_x + (trough_x - peak_x) / 2
        fig.add_shape(type="line",
                      x0=midpoint_x, y0=peak_price,
                      x1=midpoint_x, y1=trough_price,
                      line=dict(color="RoyalBlue", width=2),
                      row=1, col=1)

    # -------------------- Volume --------------------
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                         marker_color='rgba(158,202,225,0.5)'),
                  row=2, col=1)

    # -------------------- MACD --------------------
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Signal', line=dict(color='orange')),
                  row=3, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Histogram'], name='Histogram', marker_color='grey'),
                  row=3, col=1)

    # -------------------- RSI --------------------
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                  row=4, col=1)
    fig.add_hline(y=70, line_dash="dot",
                  annotation_text="Overbought", annotation_position="bottom right",
                  row=4, col=1)
    fig.add_hline(y=30, line_dash="dot",
                  annotation_text="Oversold", annotation_position="top right",
                  row=4, col=1)

    # -------------------- Layout --------------------
    fig.update_layout(title=f'{ticker} Stock Analysis (Last 1 Year)',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      height=1000)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <ticker> <threshold>")
        sys.exit(1)

    ticker = sys.argv[1]
    threshold = float(sys.argv[2])

    data = yf.download(ticker, period='1y')

    # MACD
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)

    # RSI
    data['RSI'] = calculate_rsi(data)

    # Find peaks/troughs
    peaks, troughs = find_peaks_troughs(data, threshold)

    # Plot
    plot_data(data, ticker, peaks, troughs)
