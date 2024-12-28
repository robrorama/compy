import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection

# Set the default style for seaborn
sns.set(style="whitegrid")

def aggregate_open_interest(ticker):
    # Define the data directory based on today's date and the ticker symbol
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return None

    # Combine all DataFrames into one
    all_data = pd.concat(df_list, ignore_index=True)

    # Sum open interest by strike price
    open_interest_by_strike = all_data.groupby('strike')['openInterest'].sum().reset_index()

    # Sort by open interest in descending order
    open_interest_by_strike = open_interest_by_strike.sort_values(by='openInterest', ascending=False)

    return open_interest_by_strike

def plot_open_interest(ticker):
    open_interest_by_strike = aggregate_open_interest(ticker)
    if open_interest_by_strike is None:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(open_interest_by_strike['strike'], open_interest_by_strike['openInterest'], color='skyblue')
    plt.xlabel('Strike Price')
    plt.ylabel('Total Open Interest')
    plt.title(f'Total Open Interest by Strike Price for {ticker.upper()}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_unusual_activity(ticker):
    # Aggregate data
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        expiration_date = os.path.basename(file).split('.')[1]
        df = pd.read_csv(file)
        df['expirationDate'] = expiration_date
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return

    all_data = pd.concat(df_list, ignore_index=True)

    # Filter for high open interest or volume (top 5%)
    high_open_interest = all_data[all_data['openInterest'] > all_data['openInterest'].quantile(0.95)]
    high_volume = all_data[all_data['volume'] > all_data['volume'].quantile(0.95)]

    # Combine and remove duplicates
    unusual_activity = pd.concat([high_open_interest, high_volume]).drop_duplicates()

    if unusual_activity.empty:
        print(f"No unusual activity detected for ticker {ticker}.")
        return

    # Visualize
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=unusual_activity, x='strike', y='openInterest', hue='expirationDate', palette='viridis')
    plt.title(f'Unusual Options Activity for {ticker.upper()}')
    plt.xlabel('Strike Price')
    plt.ylabel('Open Interest')
    plt.legend(title='Expiration Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_implied_volatility_surface(ticker):
    # Aggregate data
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        expiration_date = os.path.basename(file).split('.')[1]
        df = pd.read_csv(file)
        df['expirationDate'] = pd.to_datetime(expiration_date)
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return

    all_data = pd.concat(df_list, ignore_index=True)

    # Calculate days to expiration
    all_data['daysToExpiration'] = (all_data['expirationDate'] - pd.to_datetime(today_str)).dt.days

    # Filter out invalid implied volatility values
    all_data = all_data[all_data['impliedVolatility'] > 0]

    if all_data.empty:
        print(f"No valid implied volatility data for ticker {ticker}.")
        return

    # Create pivot table
    pivot_table = all_data.pivot_table(
        values='impliedVolatility',
        index='strike',
        columns='daysToExpiration',
        aggfunc='mean'
    )

    # Prepare data for plotting
    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    Z = pivot_table.values

    # Plotting
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Implied Volatility')
    plt.title(f'Implied Volatility Surface for {ticker.upper()}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def calculate_put_call_ratio(ticker):
    # Aggregate data
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        df = pd.read_csv(file)
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return

    all_data = pd.concat(df_list, ignore_index=True)

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

def plot_open_interest_heatmap(ticker):
    # Aggregate data
    today_str = datetime.today().strftime("%Y-%m-%d")
    data_dir = f"data/{today_str}/{ticker}/options_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for file in csv_files:
        expiration_date = os.path.basename(file).split('.')[1]
        df = pd.read_csv(file)
        df['expirationDate'] = expiration_date
        df_list.append(df)

    if not df_list:
        print(f"No data found for ticker {ticker} in directory {data_dir}.")
        return

    all_data = pd.concat(df_list, ignore_index=True)

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

    # Convert expiration dates to datetime for sorting
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns)
    heatmap_data = heatmap_data.sort_index()
    heatmap_data = heatmap_data.sort_index(axis=1)

    # Plot heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap='hot_r')
    plt.title(f'Open Interest Heatmap for {ticker.upper()}')
    plt.xlabel('Expiration Date')
    plt.ylabel('Strike Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    ticker = input("Enter the ticker symbol (e.g., AAPL): ").strip().upper()

    while True:
        print("\nSelect an option:")
        print("1. Plot total open interest by strike price")
        print("2. Plot unusual options activity")
        print("3. Plot implied volatility surface")
        print("4. Calculate put/call open interest ratio")
        print("5. Plot open interest heatmap")
        print("6. Exit")

        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            plot_open_interest(ticker)
        elif choice == '2':
            plot_unusual_activity(ticker)
        elif choice == '3':
            plot_implied_volatility_surface(ticker)
        elif choice == '4':
            calculate_put_call_ratio(ticker)
        elif choice == '5':
            plot_open_interest_heatmap(ticker)
        elif choice == '6':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a number between 1 and 6.")

if __name__ == "__main__":
    main()

