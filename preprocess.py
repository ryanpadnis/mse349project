import numpy as np
import pandas as pd
import yfinance as yf
import random

#data preprocessing/downloading from yfinance and cleaning/outputting to csv files
""" #unecessary, tested first with sp500
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
print(tickers.head())

holdings = pd.read_csv("IWM_holdings.csv")
tickers = holdings["Ticker"].to_list()
# Get the data for this tickers from yahoo finance

prices = yf.download(tickers,'2002-1-1','2022-1-1', auto_adjust=True)['Close']
prices.to_csv('RUSS_2000.CSV') #put into  csv file
"""

prices = pd.read_csv("RUSS_2000.CSV", parse_dates=['Date'])

# Set 'Date' column as the index
prices.set_index('Date', inplace=True)

# Resample to monthly frequency and calculate monthly returns

monthly_returns = prices.resample('M').last().pct_change() #to get monthly returns
monthly_returns.drop(columns = "-", inplace = True) #weird first column
print(monthly_returns)
threshold = 0.2  # Allow up to 20% NaN values

# Calculate the fraction of NaN values in each column
nan_fraction = monthly_returns.isna().mean()

# Drop columns where the fraction of NaN values exceeds the threshold
monthly_returns = monthly_returns.loc[:, nan_fraction <= threshold]
monthly_returns.fillna(0, inplace = True)


tickers = monthly_returns.columns.to_numpy()

N = len(tickers)
n = 200
sampled_integers = random.sample(range(N), n)
select_tickers = tickers[sampled_integers] #100 different assets out of the 500 selected at random
select_prices = pd.DataFrame(0.0, index = monthly_returns.index, columns = select_tickers)
for ticker in select_tickers:
    select_prices[ticker] = monthly_returns[ticker]

select_prices.to_csv('RUSS2000_monthly_returns')
print(select_prices)
#note, need to drop date column every time you redownload data, or alternatively set it to the index column


