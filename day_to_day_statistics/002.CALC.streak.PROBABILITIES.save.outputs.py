import pandas as pd
import sys
import os

def main():
    # Check if the CSV file is provided
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_csv_file")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Extract the ticker name from the CSV file name
    # Assuming the CSV file is named like 'spy.daily_data.csv'
    ticker = os.path.basename(csv_file).split('.')[0]
    
    # Extract the directory path of the CSV file
    source_directory = os.path.dirname(csv_file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # Ensure the data is sorted by date
    df.sort_index(inplace=True)
    
    # Compute the daily returns
    df['Return'] = df['Close'].pct_change()
    
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
    
    # Drop NaN values resulting from shift
    df.dropna(subset=['Next_Day_Up', 'Next_Day_Return'], inplace=True)
    
    # Group by 'Signed_Streak_Length' and compute counts and probabilities
    prob_table = df.groupby('Signed_Streak_Length')['Next_Day_Up'].value_counts().unstack(fill_value=0)
    
    # Compute total counts
    prob_table['Total'] = prob_table.sum(axis=1)
    
    # Compute probabilities
    prob_table['Prob_Next_Day_Up'] = prob_table[True] / prob_table['Total']
    prob_table['Prob_Next_Day_Down'] = prob_table[False] / prob_table['Total']
    
    # Compute average next day return and standard deviation
    mean_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].mean()
    std_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].std()
    
    # Compute average percentage move for the current streak
    mean_current_returns = df.groupby('Signed_Streak_Length')['Return'].mean()
    
    # Merge the mean and std returns into prob_table
    prob_table = prob_table.merge(mean_returns.rename('Avg_Next_Day_Return'), left_index=True, right_index=True)
    prob_table = prob_table.merge(std_returns.rename('Std_Next_Day_Return'), left_index=True, right_index=True)
    prob_table = prob_table.merge(mean_current_returns.rename('Avg_Current_Streak_Return'), left_index=True, right_index=True)
    
    # Sort the table by 'Signed_Streak_Length'
    prob_table.sort_index(inplace=True)
    
    # Reset index to include 'Signed_Streak_Length' as a column
    prob_table.reset_index(inplace=True)
    
    # Rename True/False columns for clarity
    prob_table.rename(columns={True: 'Next_Day_Up_Count', False: 'Next_Day_Down_Count'}, inplace=True)
    
    # Print the results without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Statistics based on streak lengths:")
        print(prob_table)
    
    # Construct the output file path
    output_file = os.path.join(source_directory, f'{ticker}_streak_statistics.csv')
    prob_table.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

