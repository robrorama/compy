import os
import pandas as pd

# Initialize an empty list to collect each ticker's dataframe
dataframes = []

# Path to the directory containing the .info.csv files
base_directory = '/media/mderby/BTRFS/FINANCE_SCRIPTS/000.2024_EVERYTHING_PYTHON/AUG_16_2024/data/2024-08-24/'

# Loop through all .info.csv files found
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.info.csv'):
            file_path = os.path.join(root, file)
            ticker_name = os.path.basename(root)
            
            # Load the CSV file
            df = pd.read_csv(file_path, header=None, names=['Field', 'Value'])
            
            # Transpose the dataframe
            df_transposed = df.set_index('Field').T
            
            # Add the ticker name as the first column
            df_transposed.insert(0, 'Ticker', ticker_name)
            
            # Append to the list of dataframes
            dataframes.append(df_transposed)

# Combine all dataframes into one master dataframe
mega_dataframe = pd.concat(dataframes, ignore_index=True)

# Optionally, save to a file
mega_dataframe.to_csv('mega_ticker_info.csv', index=False)

# Display the first few rows of the final dataframe
mega_dataframe.head()

