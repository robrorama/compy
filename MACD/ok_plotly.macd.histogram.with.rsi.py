import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ta
import warnings

# Suppress NotOpenSSLWarning (if desired)
warnings.filterwarnings('ignore', category=UserWarning)

# Download stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Calculate technical indicators
def calculate_indicators(data):
    data['20DMA'] = data['Close'].rolling(window=20).mean()
    data['50DMA'] = data['Close'].rolling(window=50).mean()
    
    # MACD Calculation
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD_Line'] - data['MACD_Signal']
    
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    return data

# Plot stock data with indicators using Plotly
def plot_stock_data(data, ticker):
    # Create the candlestick chart
    candlestick = go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlesticks')
    
    # Add moving averages
    ma_20 = go.Scatter(x=data.index, y=data['20DMA'], mode='lines', name='20DMA', line=dict(color='blue'))
    ma_50 = go.Scatter(x=data.index, y=data['50DMA'], mode='lines', name='50DMA', line=dict(color='red'))
    
    # Fill the area between 20DMA and 50DMA explicitly
    fill_between = []
    for i in range(1, len(data)):
        if data['20DMA'].iloc[i] > data['50DMA'].iloc[i]:
            fill_between.append(go.Scatter(
                x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.5)',
                line=dict(width=0),
                mode='lines',
                name='20DMA > 50DMA',
                showlegend=(i == 1),
                legendgroup='green_fill',
                hoverinfo='skip',
                visible=True
            ))
        elif data['20DMA'].iloc[i] < data['50DMA'].iloc[i]:
            fill_between.append(go.Scatter(
                x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                fill='toself',
                fillcolor='rgba(255, 0, 0, 0.5)',
                line=dict(width=0),
                mode='lines',
                name='20DMA < 50DMA',
                showlegend=(i == 1),
                legendgroup='red_fill',
                hoverinfo='skip',
                visible=True
            ))
    
    # Add MACD lines
    macd_line = go.Scatter(x=data.index, y=data['MACD_Line'], mode='lines', name='MACD Line')
    macd_signal = go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal')
    
    # Add MACD histogram
    macd_histogram = go.Bar(x=data.index, y=data['MACD_Histogram'], name='MACD Histogram', marker_color=['green' if x > 0 else 'red' for x in data['MACD_Histogram']])
    
    # Add RSI
    rsi = go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', yaxis='y2')
    
    # Layout configuration
    layout = go.Layout(
        title=f'{ticker} Stock Price and Indicators',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price'),
        yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0, 100]),
        template='plotly_dark'
    )
    
    # Combine all traces
    fig = go.Figure(data=[candlestick, ma_20, ma_50] + fill_between + [macd_line, macd_signal, macd_histogram, rsi], layout=layout)
    
    # Show plot
    fig.update_layout(legend_title_text='Indicators & Fills', showlegend=True)
    fig.show()

# Main execution
def main():
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # Fetch and process data
    data = fetch_stock_data(ticker, start_date, end_date)
    data = calculate_indicators(data)
    
    # Plot data
    plot_stock_data(data, ticker)

if __name__ == "__main__":
    main()
