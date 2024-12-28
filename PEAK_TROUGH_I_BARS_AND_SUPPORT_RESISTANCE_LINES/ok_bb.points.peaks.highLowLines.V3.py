import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import warnings
import numpy as np

# Suppress NotOpenSSLWarning (if desired)
warnings.filterwarnings('ignore', category=UserWarning)

# Download stock data from Yahoo Finance
def fetch_stock_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date)

# Calculate technical indicators
def calculate_indicators(data):
    data['20DMA'] = data['Close'].rolling(window=20).mean()
    data['50DMA'] = data['Close'].rolling(window=50).mean()
    data['9DMA'] = data['Close'].rolling(window=9).mean()
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper_1std'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 1)
    data['BB_Upper_2std'] = data['BB_Middle'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_Lower_1std'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 1)
    data['BB_Lower_2std'] = data['BB_Middle'] - (data['Close'].rolling(window=20).std() * 2)
    return data

# Plot stock data with indicators using Plotly
def plot_stock_data(data, ticker):
    # Create the candlestick chart
    candlestick = go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Candlesticks'
    )
    
    # Add moving averages
    ma_20 = go.Scatter(
        x=data.index, 
        y=data['20DMA'], 
        mode='lines', 
        name='20DMA', 
        line=dict(color='blue')
    )
    ma_50 = go.Scatter(
        x=data.index, 
        y=data['50DMA'], 
        mode='lines', 
        name='50DMA', 
        line=dict(color='red')
    )
    ma_9 = go.Scatter(
        x=data.index, 
        y=data['9DMA'], 
        mode='lines', 
        name='9DMA', 
        line=dict(color='purple')
    )
    
    # Add Bollinger Bands
    bb_middle = go.Scatter(
        x=data.index, 
        y=data['BB_Middle'], 
        mode='lines', 
        name='BB_Middle', 
        line=dict(color='gray', dash='dash')
    )
    bb_upper_1std = go.Scatter(
        x=data.index, 
        y=data['BB_Upper_1std'], 
        mode='lines', 
        name='BB_Upper_1std', 
        line=dict(color='green', dash='dot')
    )
    bb_upper_2std = go.Scatter(
        x=data.index, 
        y=data['BB_Upper_2std'], 
        mode='lines', 
        name='BB_Upper_2std', 
        line=dict(color='lightgreen', dash='dot')
    )
    bb_lower_1std = go.Scatter(
        x=data.index, 
        y=data['BB_Lower_1std'], 
        mode='lines', 
        name='BB_Lower_1std', 
        line=dict(color='orange', dash='dot')
    )
    bb_lower_2std = go.Scatter(
        x=data.index, 
        y=data['BB_Lower_2std'], 
        mode='lines', 
        name='BB_Lower_2std', 
        line=dict(color='pink', dash='dot')
    )
    
    # Add dots for open and close prices
    open_dots = go.Scatter(
        x=data.index, 
        y=data['Open'], 
        mode='markers', 
        name='Open', 
        marker=dict(color='cyan', size=2)
    )
    close_dots = go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='markers', 
        name='Close', 
        marker=dict(color='white', size=2)
    )
    
    # Add dots for high and low prices
    high_dots = go.Scatter(
        x=data.index, 
        y=data['High'], 
        mode='markers', 
        name='High', 
        marker=dict(color='green', size=2)
    )
    low_dots = go.Scatter(
        x=data.index, 
        y=data['Low'], 
        mode='markers', 
        name='Low', 
        marker=dict(color='yellow', size=2)
    )
    
    # Add dots for midpoint prices
    midpoint_dots = go.Scatter(
        x=data.index, 
        y=(data['High'] + data['Low']) / 2, 
        mode='markers', 
        name='Midpoint', 
        marker=dict(color='orange', size=2)
    )
    
    # **New Section: Conditional Dots for High/Low beyond 2std Bollinger Bands**
    high_beyond_2std_upper = go.Scatter(
        x=data.index[(data['High'] > data['BB_Upper_2std'])], 
        y=data['High'][(data['High'] > data['BB_Upper_2std'])], 
        mode='markers', 
        name='High > 2std Upper', 
        marker=dict(color='lawngreen', size=8, line=dict(color='lawngreen', width=2))
    )
    low_below_2std_lower = go.Scatter(
        x=data.index[(data['Low'] < data['BB_Lower_2std'])], 
        y=data['Low'][(data['Low'] < data['BB_Lower_2std'])], 
        mode='markers', 
        name='Low < 2std Lower', 
        marker=dict(color='red', size=8, line=dict(color='red', width=2))
    )
    
    # Fill the area between 20DMA and 50DMA
    fill_between = []
    for i in range(1, len(data)):
        if data['20DMA'].iloc[i] is not None and data['50DMA'].iloc[i] is not None:
            if data['20DMA'].iloc[i] > data['50DMA'].iloc[i]:
                fill_between.append(go.Scatter(
                    x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                    y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                    fill='toself',
                    fillcolor='rgba(0, 255, 0, 0.5)',
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='skip'
                ))
            elif data['20DMA'].iloc[i] < data['50DMA'].iloc[i]:
                fill_between.append(go.Scatter(
                    x=[data.index[i-1], data.index[i], data.index[i], data.index[i-1]],
                    y=[data['50DMA'].iloc[i-1], data['50DMA'].iloc[i], data['20DMA'].iloc[i], data['20DMA'].iloc[i-1]],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.5)',
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Layout configuration
    layout = go.Layout(
        title=f'{ticker} Stock Price and Moving Averages',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price'),
        template='plotly_dark'
    )
    
    # Combine all traces
    #fig = go.Figure(data=[candlestick, ma_20, ma_50, ma_9, bb_middle, bb_upper_1std, bb_upper_2std, bb_lower_1std, bb_lower_2std, 
    #                      open_dots, close_dots, high_dots, low_dots, midpoint_dots, 
    #                      high_beyond_2std_upper, low_below_2std_lower] + fill_between, layout=layout)
    

    # **New Section: Horizontal Lines for Highest/Lowest points beyond 2std Bollinger Bands (last 6 months)**
    six_months_ago = data.index[-1] - pd.DateOffset(months=6)
    last_six_months_data = data[data.index > six_months_ago]

    highest_above_2std_upper = last_six_months_data['High'][(last_six_months_data['High'] > last_six_months_data['BB_Upper_2std'])].max()
    lowest_below_2std_lower = last_six_months_data['Low'][(last_six_months_data['Low'] < last_six_months_data['BB_Lower_2std'])].min()

    if not np.isnan(highest_above_2std_upper):
        highest_line = go.Scatter(
            x=[data.index[0], data.index[-1]], 
            y=[highest_above_2std_upper, highest_above_2std_upper], 
            mode='lines', 
            name='Highest Above 2std Upper (Last 6M)', 
            line=dict(color='lawngreen', dash='dash', width=1)
        )
    else:
        highest_line = None

    if not np.isnan(lowest_below_2std_lower):
        lowest_line = go.Scatter(
            x=[data.index[0], data.index[-1]], 
            y=[lowest_below_2std_lower, lowest_below_2std_lower], 
            mode='lines', 
            name='Lowest Below 2std Lower (Last 6M)', 
            line=dict(color='red', dash='dash', width=1)
        )
    else:
        lowest_line = None


    # Combine all traces
    traces = [candlestick, ma_20, ma_50, ma_9, bb_middle, bb_upper_1std, bb_upper_2std, bb_lower_1std, bb_lower_2std, 
              open_dots, close_dots, high_dots, low_dots, midpoint_dots, 
              high_beyond_2std_upper, low_below_2std_lower] + fill_between

    if highest_line:
        traces.append(highest_line)
    if lowest_line:
        traces.append(lowest_line)

    fig = go.Figure(data=traces, layout=layout)  

    # Show plot
    fig.update_layout(legend_title_text='Indicators', showlegend=True)
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
