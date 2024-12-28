import pandas as pd

def get_djia_tickers():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    tables = pd.read_html(url)  # This parses all the tables in webpages to a list
    for table in tables:
        if 'Symbol' in table.columns:
            djia_table = table
            break
    return djia_table

# Fetch the data
djia_companies = get_djia_tickers()

# Optional: Set the display option to show more rows (if you want to print to screen)
pd.set_option('display.max_rows', None)

# Print the DataFrame to the screen
print(djia_companies[['Symbol', 'Company']])

# Save to a CSV file
djia_companies.to_csv('djia.csv', index=False)

