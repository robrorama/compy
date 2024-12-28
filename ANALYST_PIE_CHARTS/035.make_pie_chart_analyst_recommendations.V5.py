import os
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import pandas as pd
from datetime import datetime

# Define the stock ticker
ticker = sys.argv[1]

# Function to download data and save to CSV
def download_data(ticker):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}_recommendations.csv"

    try:
        recommendations = pd.read_csv(data_file, index_col='Datetime')
        print(f"Recommendations for {ticker} found on disk.")
    except FileNotFoundError:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        recommendations.to_csv(data_file)
        print(f"Recommendations for {ticker} downloaded and saved to disk.")
    return recommendations

# Get recommendations
recommendations = download_data(ticker)

# Count recommendations
buy_count = recommendations['buy'].sum() + recommendations['strongBuy'].sum()
hold_count = recommendations['hold'].sum()
sell_count = recommendations['sell'].sum() + recommendations['strongSell'].sum()

# Prepare data for pie chart
labels = ["Buy", "Hold", "Sell"]
sizes = [buy_count, hold_count, sell_count]

# Create pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
plt.title(f"Analyst Recommendations for {ticker}")
plt.axis("equal")

# Save the plot as a PNG file with high resolution
image_dir = f"images/{ticker}"
os.makedirs(image_dir, exist_ok=True)
image_file = os.path.join(image_dir, f'{ticker}_analyst_recommendations.png')
plt.savefig(image_file, dpi=300)

# Save a copy of the image to the PNGS folder with ticker information and date
pngs_dir = "PNGS"
os.makedirs(pngs_dir, exist_ok=True)
pngs_file = os.path.join(pngs_dir, f'{ticker}_{datetime.today().strftime("%Y-%m-%d")}_analyst_recommendations.png')
plt.savefig(pngs_file, dpi=300)

# Show the chart
plt.show()

