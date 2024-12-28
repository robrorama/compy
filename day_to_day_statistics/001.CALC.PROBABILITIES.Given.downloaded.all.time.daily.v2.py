import pandas as pd
import sys
import os

def main():
    # Check if the CSV file is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_csv_file")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if the file exists
    if not os.path.isfile(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        print("Error: 'Close' column not found in the CSV file.")
        sys.exit(1)
    
    # Ensure the index is a DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Error: The index of the DataFrame is not a DateTimeIndex. Please ensure the first column is dates.")
        sys.exit(1)
    
    # Sort the DataFrame by date
    df.sort_index(inplace=True)
    
    # Compute the daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Drop the first row with NaN return
    df.dropna(subset=['Return'], inplace=True)
    
    # Determine whether each day is an up day (True) or down day (False)
    df['Up'] = df['Return'] > 0
    
    # Compute the direction (+1 for up, -1 for down)
    df['Direction'] = df['Up'].apply(lambda x: 1 if x else -1)
    
    # Identify where the direction changes
    df['Dir_Change'] = df['Direction'] != df['Direction'].shift(1)
    
    # Compute 'Streak_ID' as the cumulative sum of 'Dir_Change'
    df['Streak_ID'] = df['Dir_Change'].cumsum()
    
    # Compute 'Streak_Length' as cumulative count within each 'Streak_ID' + 1
    df['Streak_Length'] = df.groupby('Streak_ID').cumcount() + 1
    
    # Compute 'Signed_Streak_Length' as 'Streak_Length' * 'Direction'
    df['Signed_Streak_Length'] = df['Streak_Length'] * df['Direction']
    
    # Shift 'Up' and 'Return' columns by -1 to get 'Next_Day_Up' and 'Next_Day_Return'
    df['Next_Day_Up'] = df['Up'].shift(-1)
    df['Next_Day_Return'] = df['Return'].shift(-1)
    
    # Drop the last row which will have NaN after shift
    df.dropna(subset=['Next_Day_Up', 'Next_Day_Return'], inplace=True)
    
    # Ensure 'Signed_Streak_Length' is not zero (optional, based on data)
    df = df[df['Signed_Streak_Length'] != 0]
    
    # Group by 'Signed_Streak_Length' and compute counts and probabilities
    prob_table = df.groupby('Signed_Streak_Length')['Next_Day_Up'].value_counts().unstack(fill_value=0)
    
    # Ensure both True and False columns exist
    if True not in prob_table.columns:
        prob_table[True] = 0
    if False not in prob_table.columns:
        prob_table[False] = 0
    
    # Compute total counts
    prob_table['Total'] = prob_table.get(True, 0) + prob_table.get(False, 0)
    
    # Avoid division by zero
    prob_table['Prob_Next_Day_Up'] = prob_table.get(True, 0) / prob_table['Total']
    prob_table['Prob_Next_Day_Down'] = prob_table.get(False, 0) / prob_table['Total']
    
    # Compute average next day return and standard deviation
    mean_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].mean()
    std_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].std()
    
    # Merge the mean and std returns into prob_table
    prob_table = prob_table.merge(mean_returns.rename('Avg_Next_Day_Return'), left_index=True, right_index=True)
    prob_table = prob_table.merge(std_returns.rename('Std_Next_Day_Return'), left_index=True, right_index=True)
    
    # Sort the table by 'Signed_Streak_Length'
    prob_table.sort_index(inplace=True)
    
    # Reset index for better readability (optional)
    prob_table.reset_index(inplace=True)
    
    # Display the results
    print("Statistics based on streak lengths:")
    print(prob_table[['Signed_Streak_Length', 'Total', 'Prob_Next_Day_Up', 'Prob_Next_Day_Down', 'Avg_Next_Day_Return', 'Std_Next_Day_Return']])
    
    # Optionally, save the results to a CSV file
    output_file = 'streak_statistics.csv'
    try:
        prob_table.to_csv(output_file, index=False)
        print(f"Results saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    main()
