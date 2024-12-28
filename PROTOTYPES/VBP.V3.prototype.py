import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import mplfinance as mpf

def plot_stock_with_vbp(ticker, start_date, end_date, num_bins=20):
    """
    Downloads stock data, calculates VBP, and plots a candlestick chart with VBP bars.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).
        num_bins (int): Number of bins for VBP histogram.
    """

    # Download stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate price range for binning
    min_price = data['Low'].min()
    max_price = data['High'].max()
    price_bins = np.linspace(min_price, max_price, num_bins + 1)

    # Calculate VBP
    vbp = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (data['Low'] <= price_bins[i + 1]) & (data['High'] >= price_bins[i])
        vbp[i] = data.loc[mask, 'Volume'].sum()

    # Normalize VBP
    vbp_normalized = vbp / vbp.max()

    # Create figure and gridspec
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])

    # Candlestick chart
    ax1 = plt.subplot(gs[0, 0])
    # Pass the ax explicitly and set volume to False since we have a dedicated volume subplot
    mpf.plot(data, type='candle', style='yahoo', ax=ax1, volume=False)  # No title here
    ax1.set_ylabel('Price')

    # Volume chart
    ax3 = plt.subplot(gs[1, 0], sharex=ax1)
    mpf.plot(data, type='candle', style='yahoo', ax=ax3, volume=ax3)
    ax3.set_xlabel('Date')  # Add x-axis label for the volume chart

    # VBP chart
    ax2 = plt.subplot(gs[:, 1], sharey=ax1)
    ax2.barh(price_bins[:-1], vbp_normalized, height=(price_bins[1] - price_bins[0]), align='edge', color='blue', alpha=0.4)
    ax2.set_xlabel('Volume by Price (Normalized)')
    ax2.yaxis.tick_right()
    plt.setp(ax2.get_yticklabels(), visible=False)

    # Add title to the entire figure
    fig.suptitle(f'{ticker} Stock Chart with Volume by Price ({start_date} to {end_date})')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Example usage:
ticker_symbol = 'MSFT'
start_date = '2023-01-01'
end_date = '2023-10-26'

plot_stock_with_vbp(ticker_symbol, start_date, end_date)
