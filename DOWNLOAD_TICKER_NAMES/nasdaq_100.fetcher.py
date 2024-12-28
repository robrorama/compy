import pandas as pd

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
    tables = pd.read_html(url)  # This parses all the tables in webpages to a list
    for table in tables:
        if 'Ticker' in table.columns:
            nasdaq100_table = table
            break
    return nasdaq100_table

# Fetch the data
nasdaq100_companies = get_nasdaq100_tickers()

# Optional: Set the display option to show more rows (if you want to print to screen)
pd.set_option('display.max_rows', None)

# Print the DataFrame to the screen
print(nasdaq100_companies[['Ticker', 'Company']])

# Save to a CSV file
nasdaq100_companies.to_csv('nasdaq100.csv', index=False)

