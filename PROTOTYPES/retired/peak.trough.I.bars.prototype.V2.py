import sys
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

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

def find_approx_peaks_troughs(dataframe, threshold=0.04, column='Smoothed_Close'):
    """
    Use a smoothed price series (dataframe[column]) to alternate between searching 
    for a local peak and then a local trough. Only record a new extremum if its price 
    differs from the last recorded extremum by more than the specified threshold fraction.

    Returns a list of approximate peaks and troughs in the format: 
    peaks -> list of (index_in_dataframe, approximate_peak_price)
    troughs -> list of (index_in_dataframe, approximate_trough_price)
    """
    peaks = []
    troughs = []
    searching_for_peak = True
    last_extremum_price = None

    smoothed_values = dataframe[column].values
    for i in range(1, len(smoothed_values) - 1):
        if searching_for_peak:
            # Look for local maximum on the smoothed data
            if smoothed_values[i] > smoothed_values[i - 1] and smoothed_values[i] > smoothed_values[i + 1]:
                if (last_extremum_price is None or
                   (smoothed_values[i] - last_extremum_price) / last_extremum_price > threshold):
                    peaks.append((i, smoothed_values[i]))
                    last_extremum_price = smoothed_values[i]
                    searching_for_peak = False
        else:
            # Look for local minimum on the smoothed data
            if smoothed_values[i] < smoothed_values[i - 1] and smoothed_values[i] < smoothed_values[i + 1]:
                if (last_extremum_price is None or
                   (last_extremum_price - smoothed_values[i]) / last_extremum_price > threshold):
                    troughs.append((i, smoothed_values[i]))
                    last_extremum_price = smoothed_values[i]
                    searching_for_peak = True

    return peaks, troughs

def refine_extrema_with_raw_data(dataframe, approximate_extrema, search_window=2, is_peak=True):
    """
    Given approximate extrema indices (from smoothed data) and a search_window, 
    look around each approximate index in the *raw* Close prices to find the actual 
    highest or lowest point.

    If is_peak=True, we search for the maximum in the raw data around the approximate index.
    If is_peak=False, we search for the minimum in the raw data around the approximate index.

    Returns a list of (final_index, final_price) in the raw data.
    """
    refined = []
    raw_close = dataframe['Close'].values
    for (approx_index, _) in approximate_extrema:
        left_bound = max(0, approx_index - search_window)
        right_bound = min(len(raw_close) - 1, approx_index + search_window)
        if is_peak:
            # Find maximum in that range
            local_slice = raw_close[left_bound:right_bound+1]
            local_max_price = local_slice.max()
            local_max_position = local_slice.argmax()
            final_index = left_bound + local_max_position
            final_price = local_max_price
        else:
            # Find minimum in that range
            local_slice = raw_close[left_bound:right_bound+1]
            local_min_price = local_slice.min()
            local_min_position = local_slice.argmin()
            final_index = left_bound + local_min_position
            final_price = local_min_price

        refined.append((final_index, final_price))
    return refined

def plot_data(dataframe, ticker, peaks, troughs):
    """
    Plot candlestick chart, volume, MACD, RSI, and peak-trough I-bars.
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
    # Pair each peak with the next trough in chronological order
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
            j += 1
        else:
            j += 1

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
    if len(sys.argv) != 5:
        print("Usage: python script.py <ticker> <threshold> <smoothing_days> <search_window>")
        sys.exit(1)

    ticker = sys.argv[1]
    threshold = float(sys.argv[2])
    smoothing_days = int(sys.argv[3])
    search_window = int(sys.argv[4])

    # Download one year of data
    dataframe = yf.download(ticker, period='1y')

    # Calculate MACD
    dataframe['MACD'], dataframe['Signal'], dataframe['Histogram'] = calculate_macd(dataframe)

    # Calculate RSI
    dataframe['RSI'] = calculate_rsi(dataframe)

    # Compute smoothed close prices for detection
    dataframe['Smoothed_Close'] = dataframe['Close'].rolling(window=smoothing_days).mean()

    # Drop any rows where Smoothed_Close is not yet available
    dataframe.dropna(subset=['Smoothed_Close'], inplace=True)

    # Find approximate peaks and troughs on smoothed data
    approximate_peaks, approximate_troughs = find_approx_peaks_troughs(
        dataframe,
        threshold=threshold,
        column='Smoothed_Close'
    )

    # Refine approximate peaks on raw data
    refined_peaks = refine_extrema_with_raw_data(
        dataframe,
        approximate_peaks,
        search_window=search_window,
        is_peak=True
    )

    # Refine approximate troughs on raw data
    refined_troughs = refine_extrema_with_raw_data(
        dataframe,
        approximate_troughs,
        search_window=search_window,
        is_peak=False
    )

    # Plot the results
    plot_data(dataframe, ticker, refined_peaks, refined_troughs)
