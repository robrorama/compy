import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from fredapi import Fred
import os

# Read the FRED API key from file
with open('fred.api.key.txt', 'r') as file:
    api_key = file.read().strip()

# Initialize Fred with the API key
fred = Fred(api_key=api_key)

# Define the directory to save the fetched data
data_dir = 'fred_data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Data series IDs and labels
series_ids = {
    'GDP': 'GDP',
    'Inflation (CPI)': 'CPIAUCSL',
    'Unemployment Rate': 'UNRATE',
    'Effective Fed Funds Rate': 'FEDFUNDS',
    'University of Michigan: Consumer Sentiment': 'UMCSENT',
    'S&P 500': 'SP500',
    '10-Year Treasury Yield': 'DGS10',
    'Baa Corporate Bond Yield': 'DBAA',
    'CBOE Volatility Index: VIX': 'VIXCLS',
    'Gold Price': 'GOLDAMGBD228NLBM',
    'Industrial Production Index': 'INDPRO',
    'Capacity Utilization: Total Industry': 'TCU',
    'All Employees: Manufacturing': 'MANEMP',
    'Housing Starts: Total New Privately Owned Housing Units Started': 'HOUST',
    'Existing Home Sales': 'EXHOSLUSM495S',
    'All-Transactions House Price Index for the United States': 'CSUSHPISA',
    'All Employees: Total Nonfarm': 'PAYEMS',
    'Average Hourly Earnings of All Employees: Total Private': 'CES0500000003',
    'Labor Force Participation Rate': 'CIVPART',
    'Goods and Services Trade Balance': 'BOPGSTB',
    'U.S. / Euro Foreign Exchange Rate': 'DEXUSEU',
    'Consumer Confidence Index': 'CSCICP03USM665S',
    'Advance Retail Sales: Total': 'RSAFS',
    'Producer Price Index by Commodity: All Commodities': 'PPIACO',
    'Dow Jones Industrial Average': 'DJIA',
    'NASDAQ Composite Index': 'NASDAQCOM',
    '3-Month Treasury Bill: Secondary Market Rate': 'DTB3',
    '5-Year Treasury Constant Maturity Rate': 'DGS5',
    '30-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE30US',
    'TED Spread': 'TEDRATE',
    'Real Gross Domestic Product': 'GDPC1',
    'Nonfarm Business Sector: Output Per Hour of All Persons': 'OPHNFB',
    'New Orders for Consumer Goods': 'ACOGNO',
    'Personal Consumption Expenditures: Chain-type Price Index': 'PCECTPI',
    'Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)': 'PCEPILFE',
    'Median Sales Price of Houses Sold for the United States': 'MSPUS',
    'Crude Oil Prices: West Texas Intermediate (WTI) - Cushing, Oklahoma': 'DCOILWTICO',
    'U.S. Natural Gas Price': 'GASPRICE',
    'Unemployment Rate: 25-54 years': 'LNS12300060',
    'Unemployment by Duration of Unemployment - Mean Duration in Weeks': 'UEMPMEAN',
    'Job Openings: Total Nonfarm': 'JTSJOL',
    'Quits: Total Nonfarm': 'JTSQUR',
    'M2 Money Stock': 'M2SL',
    'Total Consumer Credit Owned and Securitized, Outstanding': 'TOTALSL',
    'Commercial and Industrial Loans, All Commercial Banks': 'BUSLOANS',
    'U.S. / Chinese Yuan Foreign Exchange Rate': 'DEXCHUS',
    'U.S. Imports of Goods from China': 'IMPCH',
    'U.S. Exports of Goods to China': 'EXPCH',
    'New Privately-Owned Housing Units Completed': 'PERMIT',
    'Total Vehicle Sales': 'TOTALSA',
    'Business Inventories to Sales Ratio': 'ISRATIO',
    'Manufacturers New Orders: Durable Goods': 'DGORDER'
}

series_idsOrig = {
    'GDP': 'GDP',
    'Inflation (CPI)': 'CPIAUCSL',
    'Unemployment Rate': 'UNRATE',
    'Effective Fed Funds Rate': 'FEDFUNDS',
    'University of Michigan: Consumer Sentiment': 'UMCSENT',
    'S&P 500': 'SP500',
    '10-Year Treasury Yield': 'DGS10',
    'Baa Corporate Bond Yield': 'DBAA',
    'CBOE Volatility Index: VIX': 'VIXCLS',
    'Gold Price': 'GOLDAMGBD228NLBM',
    'Industrial Production Index': 'INDPRO',
    'Capacity Utilization: Total Industry': 'TCU',
    'All Employees: Manufacturing': 'MANEMP',
    'Housing Starts: Total New Privately Owned Housing Units Started': 'HOUST',
    'Existing Home Sales': 'EXHOSLUSM495S',
    'All-Transactions House Price Index for the United States': 'CSUSHPISA',
    'All Employees: Total Nonfarm': 'PAYEMS',
    'Average Hourly Earnings of All Employees: Total Private': 'CES0500000003',
    'Labor Force Participation Rate': 'CIVPART',
    'Goods and Services Trade Balance': 'BOPGSTB',
    'U.S. / Euro Foreign Exchange Rate': 'DEXUSEU',
    'Consumer Confidence Index': 'CSCICP03USM665S',
    'Advance Retail Sales: Total': 'RSAFS',
    'Producer Price Index by Commodity: All Commodities': 'PPIACO'
}

# Load data from individual CSV files if they exist, otherwise fetch from FRED
df = pd.DataFrame()
for label, series_id in series_ids.items():
    file_path = os.path.join(data_dir, f"{series_id}.csv")
    if os.path.exists(file_path):
        print(f"Loading {label} from {file_path}")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        try:
            print(f"Downloading {label} ({series_id}) from FRED")
            data = fred.get_series(series_id)
            data.to_csv(file_path, header=True)
        except Exception as e:
            print(f"Failed to download {label} ({series_id}): {e}")
            continue
    if not data.empty:
        df[label] = data

# Handle potential NaN values and normalize
df.fillna(0, inplace=True)
for col in df.columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Create interactive plot with Plotly
fig = go.Figure()

for col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col))

fig.update_layout(
    title='Normalized Economic Indicators (FRED)',
    xaxis_title='Year',
    yaxis_title='Normalized Value',
    legend_title='Indicators',
    hovermode='x unified'
)

fig.show()

