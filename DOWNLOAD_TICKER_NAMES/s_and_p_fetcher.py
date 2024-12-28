import pandas as pd

# Function to fetch S&P 500 tickers
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)  # This parses all the tables in webpages to a list
    sp500_table = tables[0]  # Assuming that the first table is the S&P 500 list
    return sp500_table

# Fetch the data
sp500_companies = get_sp500_tickers()

# Optional: Set the display option to show more rows (if you want to print to screen)
pd.set_option('display.max_rows', None)

# Print the DataFrame to the screen
print(sp500_companies[['Symbol', 'Security', 'GICS Sector']])

# Save to a CSV file
sp500_companies.to_csv('s_and_p.csv', index=False)

