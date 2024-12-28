import sys
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

def plot_data(data, ticker, top10, bottom10):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        ),
        row=1,
        col=1
    )

    # Plot top 10 closes (green markers)
    fig.add_trace(
        go.Scatter(
            x=top10.index,
            y=top10['Close'],
            mode='markers',
            name='Top 10 Closes',
            marker=dict(color='lime', size=10)
        ),
        row=1,
        col=1
    )

    # Plot bottom 10 closes (red markers)
    fig.add_trace(
        go.Scatter(
            x=bottom10.index,
            y=bottom10['Close'],
            mode='markers',
            name='Bottom 10 Closes',
            marker=dict(color='red', size=10)
        ),
        row=1,
        col=1
    )

    # Volume
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.5)'
        ),
        row=2,
        col=1
    )

    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            name='MACD',
            line=dict(color='blue')
        ),
        row=3,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Signal'],
            name='Signal',
            line=dict(color='orange')
        ),
        row=3,
        col=1
    )
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Histogram'],
            name='Histogram',
            marker_color='grey'
        ),
        row=3,
        col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['RSI'],
            name='RSI',
            line=dict(color='purple')
        ),
        row=4,
        col=1
    )
    fig.add_hline(
        y=70,
        line_dash="dot",
        annotation_text="Overbought",
        annotation_position="bottom right",
        row=4,
        col=1
    )
    fig.add_hline(
        y=30,
        line_dash="dot",
        annotation_text="Oversold",
        annotation_position="top right",
        row=4,
        col=1
    )

    fig.update_layout(
        title=f'{ticker} Stock Analysis (Last 1 Year)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=900
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)

    fig.show()

if __name__ == "__main__":
    """
    Usage:
        python script.py <ticker>
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <ticker>")
        sys.exit(1)

    ticker = sys.argv[1]

    # Download data
    data = yf.download(ticker, period='1y')
    if data.empty:
        print("No data returned from yfinance. Check ticker and internet connection.")
        sys.exit(1)

    # Calculate MACD, RSI
    data['MACD'], data['Signal'], data['Histogram'] = calculate_macd(data)
    data['RSI'] = calculate_rsi(data)

    # Pick top 10 highest closes, bottom 10 lowest closes
    # We sort by 'Close' ascending, take head(10) for the bottom 10, tail(10) for the top 10.
    bottom10 = data.sort_values(by='Close', ascending=True).head(10)
    top10 = data.sort_values(by='Close', ascending=True).tail(10)

    # Plot
    plot_data(data, ticker, top10, bottom10)
