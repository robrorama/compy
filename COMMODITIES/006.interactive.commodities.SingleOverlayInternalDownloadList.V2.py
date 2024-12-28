import os
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def download_data(ticker, period):
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}"
    os.makedirs(data_dir, exist_ok=True)
    data_file = f"{data_dir}/{ticker}.csv"

    try:
        df = pd.read_csv(data_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"Data for {ticker} found on disk.")
    except FileNotFoundError:
        df = yf.download(ticker, period=period)
        df.to_csv(data_file)
        print(f"Data for {ticker} downloaded and saved to disk.")
    return df

def normalize_data(df):
    return (df["Close"] - df["Close"].mean()) / df["Close"].std()

def plot_grouped_commodities(groups, period='1y'):
    fig = go.Figure()

    group_titles = ["Energy", "Metals", "Agriculture", "Livestock"]

    for group_title, (group_name, commodities) in zip(group_titles, groups.items()):
        for name, ticker in commodities.items():
            df = download_data(ticker, period)
            norm_data = normalize_data(df)
            fig.add_trace(go.Scatter(x=norm_data.index, y=norm_data, mode='lines', name=f'{group_title}: {name}'))

    fig.update_layout(
        title="Normalized Commodity Prices",
        xaxis_title="Date",
        yaxis_title="Normalized Price",
        legend_title="Commodities",
        width=1200,
        height=800
    )

    # Save high-resolution image
    image_dir = "images/commodities"
    os.makedirs(image_dir, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    image_file = os.path.join(image_dir, f'commodities_normalized_grouped_{today_str}.png')
    fig.write_image(image_file, scale=3)
    
    print(f"High-resolution image saved to {image_file}")

    fig.show()

def main():
    commodity_groups = {
        "Energy": {
            "Light Sweet Crude Oil": "CL=F",
            "Brent Crude Oil": "BZ=F",
            "Natural Gas": "NG=F",
            "Heating Oil": "HO=F",
            "RBOB Gasoline": "RB=F"
        },
        "Metals": {
            "Gold": "GC=F",
            "Silver": "SI=F",
            "Copper": "HG=F",
            "Platinum": "PL=F",
            "Palladium": "PA=F"
        },
        "Agriculture": {
            "Corn": "ZC=F",
            "Wheat": "ZW=F",
            "Soybeans": "ZS=F",
            "Soybean Oil": "ZL=F",
            "Soybean Meal": "ZM=F",
            "Sugar": "SB=F",
            "Coffee": "KC=F",
            "Cocoa": "CC=F",
            "Cotton": "CT=F",
            "Orange Juice": "OJ=F"
        },
        "Livestock": {
            "Live Cattle": "LE=F",
            "Lean Hogs": "HE=F",
            "Feeder Cattle": "GF=F"
        }
    }

    period = '1y'
    if len(sys.argv) > 1:
        period = sys.argv[1]

    plot_grouped_commodities(commodity_groups, period)

if __name__ == '__main__':
    import sys
    main()
