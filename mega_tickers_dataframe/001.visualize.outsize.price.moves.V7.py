import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sys
import os

# Step 1: Load the Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Step 2: Volatility Measure
def calculate_volatility(df):
    df['Volatility'] = df['High'] - df['Low']
    return df

# Step 3: Daily Percentage Move
def calculate_daily_percentage_move(df):
    df['Daily Percentage Move'] = df['Close'].pct_change() * 100
    return df

# Step 4: First Derivative (Rate of Change) for Price
def calculate_instantaneous_price_roc(df):
    df['Price ROC'] = df['Close'].diff()
    return df

# Step 5: First Derivative (Rate of Change) for Volume
def calculate_instantaneous_volume_roc(df):
    df['Volume ROC'] = df['Volume'].diff()
    return df

# Step 6: Second Derivative for Price
def calculate_second_derivative_price(df):
    df['Price Second Derivative'] = df['Close'].diff().diff()
    return df

# Step 7: Second Derivative for Volume
def calculate_second_derivative_volume(df):
    df['Volume Second Derivative'] = df['Volume'].diff().diff()
    return df

# Step 8: Calculate Moving Averages
def calculate_moving_averages(df):
    ma_periods = [9, 20, 50, 100, 150, 200, 300]
    for period in ma_periods:
        df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period).mean()
    return df

# Step 9: Calculate Consecutive Up/Down Days for Price
def calculate_consecutive_days(df):
    df['Consecutive Days'] = 0  # Initialize with zeros

    price_diff = df['Close'].diff()

    consecutive = 0
    for i in range(1, len(price_diff)):
        if price_diff[i] > 0:
            consecutive = consecutive + 1 if consecutive >= 0 else 1
        elif price_diff[i] < 0:
            consecutive = consecutive - 1 if consecutive <= 0 else -1
        else:
            consecutive = 0
        df.at[i, 'Consecutive Days'] = consecutive

    return df

# Step 10: Calculate Consecutive Up/Down Days for Volume
def calculate_consecutive_volume_days(df):
    df['Consecutive Volume Days'] = 0  # Initialize with zeros

    volume_diff = df['Volume'].diff()

    consecutive = 0
    for i in range(1, len(volume_diff)):
        if volume_diff[i] > 0:
            consecutive = consecutive + 1 if consecutive >= 0 else 1
        elif volume_diff[i] < 0:
            consecutive = consecutive - 1 if consecutive <= 0 else -1
        else:
            consecutive = 0
        df.at[i, 'Consecutive Volume Days'] = consecutive

    return df

# Step 11: Calculate Sigma Moves for Daily Percentage Move
def calculate_sigma_moves(df):
    df['Sigma'] = df['Daily Percentage Move'].rolling(window=20).std()

    # Round to the closest sigma level
    def assign_sigma_level(row):
        if pd.isna(row['Daily Percentage Move']) or pd.isna(row['Sigma']):
            return None
        sigma_value = abs(row['Daily Percentage Move']) / row['Sigma']
        if sigma_value >= 3:
            return '3+'
        elif sigma_value >= 2:
            return '2'
        elif sigma_value >= 1:
            return '1'
        else:
            return None

    df['Sigma Level'] = df.apply(assign_sigma_level, axis=1)
    return df

# Plotting Functions
def plot_volatility(df):
    fig = px.line(df, x='Date', y='Volatility', title='Daily Volatility')
    fig.show()

def plot_daily_percentage_move(df):
    fig = go.Figure()

    # Plot the daily percentage move line
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Daily Percentage Move'], mode='lines', name='Daily Percentage Move'))

    # Plot sigma-level dots
    sigma_colors = {'1': 'yellow', '2': 'orange', '3+': 'red'}

    for sigma_level, color in sigma_colors.items():
        sigma_df = df[df['Sigma Level'] == sigma_level]
        fig.add_trace(go.Scatter(
            x=sigma_df['Date'], y=sigma_df['Daily Percentage Move'],
            mode='markers', marker=dict(color=color, size=8),
            name=f'{sigma_level} Sigma Move'
        ))

    # Highlight anything above 3 sigma with green dots
    green_df = df[df['Sigma Level'] == '3+']
    if not green_df.empty:
        fig.add_trace(go.Scatter(
            x=green_df['Date'], y=green_df['Daily Percentage Move'],
            mode='markers', marker=dict(color='green', size=8),
            name='3+ Sigma Move'
        ))

    fig.update_layout(title='Daily Percentage Move with Sigma Levels', xaxis_title='Date', yaxis_title='Percentage Move')
    fig.show()

def plot_price_derivatives(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price ROC'], mode='lines', name='Price ROC'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Price Second Derivative'], mode='lines', name='Price Second Derivative'))
    fig.update_layout(title='First and Second Derivative of Price', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def plot_volume_derivatives(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume ROC'], mode='lines', name='Volume ROC'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume Second Derivative'], mode='lines', name='Volume Second Derivative'))
    fig.update_layout(title='First and Second Derivative of Volume', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def plot_daily_price_with_moving_averages(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Daily Price'))

    ma_periods = [9, 20, 50, 100, 150, 200, 300]
    for period in ma_periods:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'MA_{period}'], mode='lines', name=f'MA {period}'))

    fig.update_layout(title='Daily Price with Moving Averages', xaxis_title='Date', yaxis_title='Price')
    fig.show()

def plot_daily_volume_with_moving_averages(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume'], mode='lines', name='Daily Volume'))

    ma_periods = [9, 20, 50, 100, 150, 200, 300]
    for period in ma_periods:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'Volume_MA_{period}'], mode='lines', name=f'Volume MA {period}'))

    fig.update_layout(title='Daily Volume with Moving Averages', xaxis_title='Date', yaxis_title='Volume')
    fig.show()

# Plot: Consecutive Up/Down Days for Price
def plot_consecutive_days(df):
    fig = px.bar(df, x='Date', y='Consecutive Days', title='Consecutive Up/Down Days for Price',
                 labels={'Consecutive Days': 'Consecutive Days (Up/Down)'})
    fig.update_traces(marker_color=['green' if x > 0 else 'red' for x in df['Consecutive Days']])
    fig.show()

# Plot: Consecutive Up/Down Days for Volume
def plot_consecutive_volume_days(df):
    fig = px.bar(df, x='Date', y='Consecutive Volume Days', title='Consecutive Up/Down Days for Volume',
                 labels={'Consecutive Volume Days': 'Consecutive Volume Days (Up/Down)'})
    fig.update_traces(marker_color=['green' if x > 0 else 'red' for x in df['Consecutive Volume Days']])
    fig.show()

# Placeholder: Other Calculations
def plot_other_metrics(df):
    # Placeholder function for future plots
    pass

# Main Function to Run the Analysis
def main(file_path):
    df = load_data(file_path)

    # Calculate Volatility
    df = calculate_volatility(df)

    # Calculate Daily Percentage Move
    df = calculate_daily_percentage_move(df)

    # Calculate First and Second Derivatives for Price and Volume
    df = calculate_instantaneous_price_roc(df)
    df = calculate_instantaneous_volume_roc(df)
    df = calculate_second_derivative_price(df)
    df = calculate_second_derivative_volume(df)

    # Calculate Moving Averages for Price and Volume
    df = calculate_moving_averages(df)

    # Calculate Consecutive Up/Down Days for Price
    df = calculate_consecutive_days(df)

    # Calculate Consecutive Up/Down Days for Volume
    df = calculate_consecutive_volume_days(df)

    # Placeholder for other calculations
    df = calculate_sigma_moves(df)

    # Plot Results
    plot_volatility(df)
    plot_daily_percentage_move(df)
    plot_price_derivatives(df)
    plot_volume_derivatives(df)
    plot_daily_price_with_moving_averages(df)
    plot_daily_volume_with_moving_averages(df)
    plot_consecutive_days(df)
    plot_consecutive_volume_days(df)
    plot_other_metrics(df)

    #import os
    output_dir = os.path.dirname(file_path)
    ticker_name = os.path.basename(file_path).split('.')[0]
    output_filename = f"{ticker_name}.processed_data.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Old way Save the processed DataFrame for future use
    #df.to_csv("processed_data.csv", index=False)

# Run the analysis if script is called directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)

