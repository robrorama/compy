import pandas as pd

# Load the dataset
file_path = 'your_input_file.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Ensure 'marketCap' is treated as a string first, then as a numeric value
df['marketCap'] = pd.to_numeric(df['marketCap'].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')

# Drop rows where 'marketCap' is NaN
df = df.dropna(subset=['marketCap'])

# Filter out companies with market capitalization below 1 billion
df_filtered = df[df['marketCap'] > 1_000_000_000]

# Function to sort the DataFrame based on a specified column
def sort_by_column(df, column_name):
    if column_name in df.columns:
        # Handle non-numeric columns by converting to numeric if needed
        df[column_name] = pd.to_numeric(df[column_name].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
        df_sorted = df.sort_values(by=column_name, ascending=False)
        return df_sorted
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return df

# Specify the column name to sort by (e.g., 'marketCap', 'revenueGrowth', etc.)
sort_column = 'marketCap'  # Replace with your desired column header

# Sort the DataFrame based on the specified column
df_sorted = sort_by_column(df_filtered, sort_column)

# Define the columns to include in the output
columns = [
    'marketCap',                # Primary sorting column
    'trailingPE',               # P/E Ratio (Trailing)
    'forwardPE',                # P/E Ratio (Forward)
    'returnOnEquity',           # ROE
    'priceToBook',              # Price-to-Book Ratio
    'priceToSalesTrailing12Months',  # Price-to-Sales Ratio
    'returnOnAssets',           # ROA
    'operatingCashflow',        # Operating Cash Flow
    'revenueGrowth',            # Revenue Growth
    'grossMargins',             # Gross Margins
    'operatingMargins',         # Operating Margins
    'dividendYield',            # Dividend Yield
    'earningsGrowth',           # Earnings Growth
    'enterpriseToEbitda',       # EV/EBITDA
    'beta',                     # Beta
    'pegRatio',                 # PEG Ratio
    'profitMargins',            # Profit Margins
    'totalRevenue',             # Total Revenue
    'quickRatio',               # Quick Ratio
    'currentRatio',             # Current Ratio
    'Ticker'                    # Ticker
]

# Select only the columns of interest
df_sorted = df_sorted[columns]

# Save the sorted DataFrame to a new CSV file
output_file_path = 'sorted_stock_data.csv'
df_sorted.to_csv(output_file_path, index=False)

print(f"Sorted data saved to {output_file_path}")

