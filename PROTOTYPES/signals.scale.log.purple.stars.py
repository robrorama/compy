import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sys

def download_data(ticker, start_date, end_date):
    """Downloads stock price data using yfinance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"Error: No data found for ticker '{ticker}'. Please check the ticker symbol.")
            return None
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def calculate_indicators(data):
    """Calculates log price, ROC, moving averages, volatility, and momentum."""
    data['Log_Price'] = np.log(data['Adj Close'])
    data['ROC'] = data['Adj Close'].pct_change()  # Rate of Change
    
    # Calculate moving averages
    ma_days = [9, 20, 50, 100, 200, 300]
    for ma in ma_days:
        data[f'MA_{ma}'] = data['Adj Close'].rolling(window=ma).mean()

    data['Volatility'] = data['ROC'].rolling(window=20).std() * np.sqrt(252)  # Annualized Volatility
    data['Momentum'] = data['Adj Close'] - data['Adj Close'].shift(20)  # 20-day Momentum
    return data

def power_law(x, a, b):
    """Power law function for fitting."""
    return a * np.power(x, b)

def identify_potential_bubble_patterns(data):
    """Identifies potential bubble patterns using power-law fit and log-periodic analysis (optional)."""
    data['Bubble_Signal'] = 0  # Initialize bubble signal
    data['LPPL_Signal'] = 0  # Initialize LPPL signal
    data['LPPL_Confidence'] = 0 # Initialize LPPL confidence

    # Power Law Detection:
    try:
        x_data = np.arange(len(data['Log_Price'].tail(60)))
        y_data = data['Log_Price'].tail(60).values

        params, covariance = curve_fit(power_law, x_data, y_data)

        if params[1] > 1.1:
            data.loc[data.index[-60:], 'Bubble_Signal'] = 1
    except Exception as e:
        print(f"Could not fit power-law to recent data: {e}")

    # LPPL Analysis
    try:
        # Fit LPPL to recent data (e.g., last 180 days)
        t = np.arange(len(data['Log_Price'].tail(180)))
        log_price = data['Log_Price'].tail(180).values

        # LPPL function
        def lppl(t, tc, m, omega, A, B, C, phi):
            return A + B * (tc - t)**m + C * (tc - t)**m * np.cos(omega * np.log(tc - t) - phi)

        # Initial parameter guesses
        p0 = [len(t) + 30, 0.5, 10, log_price[-1], -0.5, 0.1, 0]

        # Fit the model
        params, covariance = curve_fit(lppl, t, log_price, p0=p0, bounds=([len(t), 0, 6, -np.inf, -np.inf, -np.inf, -np.inf],[len(t) * 2, 1, 15, np.inf, 0, np.inf, np.inf]))
        
        tc, m, omega, A, B, C, phi = params

        # Calculate LPPL confidence (distance from tc to end of data)
        lppl_confidence = np.abs(tc - len(t))

        # LPPL-based bubble detection criteria
        if (len(t) < tc < len(t) * 1.5) and (0 < m < 1) and (6 < omega < 15) and (B < 0):
            data.loc[data.index[-180:], 'LPPL_Signal'] = 1
            data.loc[data.index[-180:], 'LPPL_Confidence'] = lppl_confidence # Store confidence
            print(f"Potential bubble detected (LPPL): tc={tc:.2f}, m={m:.2f}, omega={omega:.2f}, B={B:.2f}")
            # Store fitted LPPL values for plotting
            data['LPPL_Fit'] = np.nan
            data.loc[data.index[-180:], 'LPPL_Fit'] = lppl(t,*params)

    except Exception as e:
        print(f"Could not fit LPPL model to recent data: {e}")

    return data

def signal_processing(data):
    """Applies Fourier analysis to identify cyclical patterns."""
    data['Fourier_Signal'] = 0  # Initialize Fourier signal

    # Fourier Transform
    fft_values = np.fft.fft(data['Adj Close'].values)
    fft_values = np.abs(fft_values) ** 2  # Power spectrum

    # Find dominant frequencies (peaks in the power spectrum)
    peaks, _ = find_peaks(fft_values, height=np.percentile(fft_values, 95))

    # Convert peak indices to frequencies (cycles per unit time - here, per day)
    frequencies = peaks / len(data)

    # Filter out very low frequencies (long-term trends) and very high frequencies (noise)
    meaningful_frequencies = [f for f in frequencies if 0.01 < f < 0.5]

    if meaningful_frequencies:
        data['Fourier_Signal'] = 1
        print(f"Dominant Frequencies (cycles/day): {meaningful_frequencies}")

    return data

def visualize(data, ticker):
    """Plots stock price, indicators, and highlights potential bubble patterns using Plotly."""
    
    # Create subplots
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=[f'{ticker} Stock Price and Moving Averages', 'Log Price and Bubble Signals', 'Volatility and Momentum', 'Fourier Signal', 'Percentage Price Movements with Moving Averages'])

    # Plot 1: Stock Price (Candlestick) and Moving Averages
    fig.add_trace(go.Candlestick(x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'], name='Candlestick'), row=1, col=1)

    ma_days = [9, 20, 50, 100, 200, 300]
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink']
    for ma, color in zip(ma_days, colors):
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma}'], mode='lines', name=f'{ma}-day MA', line=dict(color=color)), row=1, col=1)

    # Plot 2: Log Price and Bubble Signals
    fig.add_trace(go.Scatter(x=data.index, y=data['Log_Price'], mode='lines', name='Log Price', line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data[data['Bubble_Signal'] == 1].index, y=data[data['Bubble_Signal'] == 1]['Log_Price'], mode='markers', name='Potential Bubble (Power Law)', 
                             marker=dict(color='red', symbol='triangle-up', size=12, line=dict(color='black', width=1))), row=2, col=1)
    
    lppl_signals = data[data['LPPL_Signal'] == 1]
    if not lppl_signals.empty:
        lppl_signals = lppl_signals.sort_values('LPPL_Confidence')
        most_significant_lppl_signals = lppl_signals.tail(5)

        fig.add_trace(go.Scatter(x=most_significant_lppl_signals.index, y=most_significant_lppl_signals['Log_Price'], mode='markers', name='Potential Bubble (LPPL)', 
                                 marker=dict(color='magenta', symbol='star', size=14, line=dict(color='black', width=1))), row=2, col=1)
    
    if 'LPPL_Fit' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['LPPL_Fit'], mode='lines', name='LPPL Fit', line=dict(color='black', dash='dash')), row=2, col=1)

    # Plot 3: Volatility and Momentum
    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility', line=dict(color='brown')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Momentum'], mode='lines', name='Momentum', line=dict(color='gray')), row=3, col=1)

    # Plot 4: Fourier Signal
    fig.add_trace(go.Scatter(x=data[data['Fourier_Signal'] == 1].index, y=[1] * sum(data['Fourier_Signal'] == 1), mode='markers', name='Potential Cyclical Pattern (Fourier)', marker=dict(color='cyan', symbol='circle', size=8)), row=4, col=1)
    fig.update_yaxes(range=[0, 1.1], tickvals=[0, 1], ticktext=['No Signal', 'Signal'], row=4, col=1)

    # Plot 5: Percentage Price Movements with Moving Averages
    data['Price_Pct_Change'] = data['Adj Close'].pct_change() * 100
    fig.add_trace(go.Scatter(x=data.index, y=data['Price_Pct_Change'], mode='lines', name='Price % Change', line=dict(color='blue')), row=5, col=1)
    
    for ma, color in zip(ma_days, colors):
        data[f'MA_{ma}_Pct_Change'] = data[f'MA_{ma}'].pct_change() * 100
        fig.add_trace(go.Scatter(x=data.index, y=data[f'MA_{ma}_Pct_Change'], mode='lines', name=f'{ma}-day MA % Change', line=dict(color=color)), row=5, col=1)
    
    # Update layout
    fig.update_layout(title_text=f'{ticker} Stock Analysis', title_x=0.5, height=1400, xaxis_rangeslider_visible=False)
    
    # Annotations for first bubble signals (Power Law only)
    if not data[data['Bubble_Signal'] == 1].empty:
        first_power_law_signal_index = data[data['Bubble_Signal'] == 1].index[0]
        fig.add_annotation(x=first_power_law_signal_index, y=data.loc[first_power_law_signal_index]['Log_Price'],
                           text="First Power Law Signal", showarrow=True, arrowhead=1, arrowcolor="red", ax=0, ay=-40, row=2, col=1)

    fig.show()

# --- Main Execution ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]
    start_date = '2024-01-01'
    end_date = '2024-12-31'

    # 1. Download Data
    stock_data = download_data(ticker, start_date, end_date)
    if stock_data is None:
        exit()

    # 2. Calculate Indicators
    stock_data = calculate_indicators(stock_data)

    # 3. Identify Potential Bubble Patterns
    stock_data = identify_potential_bubble_patterns(stock_data)

    # 4. Signal Processing
    stock_data = signal_processing(stock_data)

    # 5. Visualize
    visualize(stock_data, ticker)
