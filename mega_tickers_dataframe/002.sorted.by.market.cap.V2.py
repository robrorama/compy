import pandas as pd

# Load the dataset
file_path = 'your_input_file.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Ensure 'marketCap' is treated as a string first, handling missing or non-string data
df['marketCap'] = df['marketCap'].astype(str).str.replace(',', '').str.replace('$', '')

# Convert 'marketCap' to numeric, forcing errors to NaN
df['marketCap'] = pd.to_numeric(df['marketCap'], errors='coerce')

# Drop rows where 'marketCap' is NaN
df = df.dropna(subset=['marketCap'])

# Filter out companies with market capitalization below 1 billion
df_filtered = df[df['marketCap'] > 1_000_000_000]

# Define the columns based on importance
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

# Select only the columns of interest and sort the filtered DataFrame by market capitalization
df_sorted = df_filtered.sort_values(by='marketCap', ascending=False)[columns]

# Save the sorted DataFrame to a new CSV file
output_file_path = 'sorted_stock_data.csv'
df_sorted.to_csv(output_file_path, index=False)

print(f"Sorted data saved to {output_file_path}")

