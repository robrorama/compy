import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def aggregate_data(ticker):
    # Define the data directory based on today's date and the ticker symbol
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        expiration_date = os.path.basename(file).split('.')[1]
        df = pd.read_csv(file)
        df['expirationDate'] = pd.to_datetime(expiration_date)
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return None

    # Combine all DataFrames into one
    all_data = pd.concat(df_list, ignore_index=True)
    return all_data

def plot_open_interest(ticker, all_data):
    # Sum open interest by strike price
    open_interest_by_strike = all_data.groupby('strike')['openInterest'].sum().reset_index()

    # Create interactive bar chart
    fig = px.bar(
        open_interest_by_strike,
        x='strike',
        y='openInterest',
        title=f'Total Open Interest by Strike Price for {ticker.upper()}',
        labels={'strike': 'Strike Price', 'openInterest': 'Total Open Interest'},
        template='plotly_dark'
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()

def plot_unusual_activity(ticker, all_data):
    # Filter for high open interest or volume (top 5%)
    high_open_interest = all_data[all_data['openInterest'] > all_data['openInterest'].quantile(0.95)]
    high_volume = all_data[all_data['volume'] > all_data['volume'].quantile(0.95)]

    # Combine and remove duplicates
    unusual_activity = pd.concat([high_open_interest, high_volume]).drop_duplicates()

    if unusual_activity.empty:
        print(f"No unusual activity detected for ticker {ticker}.")
        return

    # Create interactive scatter plot
    fig = px.scatter(
        unusual_activity,
        x='strike',
        y='openInterest',
        color='expirationDate',
        title=f'Unusual Options Activity for {ticker.upper()}',
        labels={'strike': 'Strike Price', 'openInterest': 'Open Interest'},
        template='plotly_dark'
    )
    fig.update_layout(legend_title_text='Expiration Date')
    fig.show()

def plot_implied_volatility_surface(ticker, all_data):
    # Calculate days to expiration
    today_str = datetime.today().strftime("%Y-%m-%d")
    all_data['daysToExpiration'] = (all_data['expirationDate'] - pd.to_datetime(today_str)).dt.days

    # Filter out invalid implied volatility values
    all_data = all_data[all_data['impliedVolatility'] > 0]

    if all_data.empty:
        print(f"No valid implied volatility data for ticker {ticker}.")
        return

    # Prepare data for surface plot
    pivot_table = all_data.pivot_table(
        values='impliedVolatility',
        index='strike',
        columns='daysToExpiration',
        aggfunc='mean'
    )
    X = pivot_table.columns
    Y = pivot_table.index
    Z = pivot_table.values

    # Create interactive surface plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=f'Implied Volatility Surface for {ticker.upper()}',
        scene=dict(
            xaxis_title='Days to Expiration',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility'
        ),
        template='plotly_dark'
    )
    fig.show()

def calculate_put_call_ratio(ticker, all_data):
    # Ensure 'contractSymbol' column exists
    if 'contractSymbol' not in all_data.columns:
        print("Column 'contractSymbol' not found in data.")
        return

    # Group by option type
    put_data = all_data[all_data['contractSymbol'].str.contains('P')]
    call_data = all_data[all_data['contractSymbol'].str.contains('C')]

    # Sum open interest
    total_put_oi = put_data['openInterest'].sum()
    total_call_oi = call_data['openInterest'].sum()

    if total_call_oi == 0:
        print("Total call open interest is zero, cannot calculate put/call ratio.")
        return

    put_call_ratio = total_put_oi / total_call_oi
    print(f"Put/Call Open Interest Ratio for {ticker.upper()}: {put_call_ratio:.2f}")

def plot_open_interest_heatmap(ticker, all_data):
    # Pivot the data
    heatmap_data = all_data.pivot_table(
        index='strike',
        columns='expirationDate',
        values='openInterest',
        aggfunc='sum',
        fill_value=0
    )

    if heatmap_data.empty:
        print(f"No data available to plot heatmap for ticker {ticker}.")
        return

    # Convert expiration dates to strings for labeling
    heatmap_data.columns = heatmap_data.columns.strftime('%Y-%m-%d')

    # Create interactive heatmap
    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale='hot',
        aspect='auto',
        labels=dict(x='Expiration Date', y='Strike Price', color='Open Interest'),
        title=f'Open Interest Heatmap for {ticker.upper()}'
    )
    fig.update_layout(template='plotly_dark')
    fig.show()

def main():
    ticker = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()
    all_data = aggregate_data(ticker)
    if all_data is None:
        return

    # Generate all visualizations
    plot_open_interest(ticker, all_data)
    plot_unusual_activity(ticker, all_data)
    plot_implied_volatility_surface(ticker, all_data)
    calculate_put_call_ratio(ticker, all_data)
    plot_open_interest_heatmap(ticker, all_data)

if __name__ == "__main__":
    main()

