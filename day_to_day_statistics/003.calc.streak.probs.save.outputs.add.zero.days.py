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
    
    # Determine the direction:
    # 1 for up days, -1 for down days, 0 for no change
    def get_direction(return_value):
        if return_value > 0:
            return 1
        elif return_value < 0:
            return -1
        else:
            return 0  # Neutral day
    
    df['Direction'] = df['Return'].apply(get_direction)
    
    # Identify where the direction changes
    df['Dir_Change'] = df['Direction'] != df['Direction'].shift(1)
    
    # Compute 'Streak_ID' as the cumulative sum of 'Dir_Change'
    df['Streak_ID'] = df['Dir_Change'].cumsum()
    
    # Compute 'Streak_Length' as cumulative count within each 'Streak_ID' + 1
    df['Streak_Length'] = df.groupby('Streak_ID').cumcount() + 1
    
    # Compute 'Signed_Streak_Length' as 'Streak_Length' * 'Direction'
    df['Signed_Streak_Length'] = df.apply(
        lambda row: row['Streak_Length'] * row['Direction'] if row['Direction'] != 0 else 0,
        axis=1
    )
    
    # Shift 'Direction' and 'Return' columns by -1 to get next day's values
    df['Next_Day_Direction'] = df['Direction'].shift(-1)
    df['Next_Day_Return'] = df['Return'].shift(-1)
    
    # Drop NaN values resulting from shift
    df.dropna(subset=['Next_Day_Direction', 'Next_Day_Return'], inplace=True)
    
    # Group by 'Signed_Streak_Length' and compute counts
    direction_counts = df.groupby('Signed_Streak_Length')['Next_Day_Direction'].value_counts().unstack(fill_value=0)
    
    # Compute total counts
    direction_counts['Total'] = direction_counts.sum(axis=1)
    
    # Compute probabilities
    for direction in [-1, 0, 1]:
        col_name = f'Prob_Next_Day_Dir_{direction}'
        direction_counts[col_name] = direction_counts.get(direction, 0) / direction_counts['Total']
    
    # Compute average next day return and standard deviation
    mean_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].mean()
    std_returns = df.groupby('Signed_Streak_Length')['Next_Day_Return'].std()
    
    # Compute average percentage move for the current streak
    mean_current_returns = df.groupby('Signed_Streak_Length')['Return'].mean()
    
    # Merge the mean and std returns into direction_counts
    stats_table = direction_counts.merge(mean_returns.rename('Avg_Next_Day_Return'), left_index=True, right_index=True)
    stats_table = stats_table.merge(std_returns.rename('Std_Next_Day_Return'), left_index=True, right_index=True)
    stats_table = stats_table.merge(mean_current_returns.rename('Avg_Current_Streak_Return'), left_index=True, right_index=True)
    
    # Sort the table by 'Signed_Streak_Length'
    stats_table.sort_index(inplace=True)
    
    # Reset index to include 'Signed_Streak_Length' as a column
    stats_table.reset_index(inplace=True)
    
    # Rename direction columns for clarity
    direction_columns = {d: f'Next_Day_Dir_{d}_Count' for d in [-1, 0, 1]}
    stats_table.rename(columns=direction_columns, inplace=True)
    
    # Print the results without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Statistics based on streak lengths (including neutral days):")
        print(stats_table)
    
    # Construct the output file path
    output_file = os.path.join(source_directory, f'{ticker}_streak_statistics_with_zeroes.csv')
    stats_table.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Additional analysis: Impact of previous day's big moves
    # Define a threshold for "big" percentage moves (e.g., more than 2%)
    threshold = 0.02  # 2%
    
    # Identify days with big moves
    df['Big_Move_Previous_Day'] = df['Return'].shift(1).abs() > threshold
    
    # Calculate probabilities based on big moves the previous day
    big_move_stats = df[df['Big_Move_Previous_Day']].groupby('Direction')['Next_Day_Direction'].value_counts().unstack(fill_value=0)
    big_move_stats['Total'] = big_move_stats.sum(axis=1)
    
    for direction in [-1, 0, 1]:
        col_name = f'Prob_Next_Day_Dir_{direction}'
        big_move_stats[col_name] = big_move_stats.get(direction, 0) / big_move_stats['Total']
    
    big_move_stats.reset_index(inplace=True)
    
    print("\nStatistics based on big percentage moves the previous day:")
    print(big_move_stats)
    
    # Save the big move stats
    big_move_output_file = os.path.join(source_directory, f'{ticker}_big_move_statistics.csv')
    big_move_stats.to_csv(big_move_output_file, index=False)
    print(f"Big move results saved to {big_move_output_file}")
    
if __name__ == "__main__":
    main()

