import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

import numpy as np
import pandas as pd
# from scipy.optimize import minimize
import os
# import seaborn as sns; sns.set(style="whitegrid")
import datetime
import yahooquery as yq
# from pathlib import Path
import wikipedia as wp
import io

# Define path for saving results
# results_path = Path('results', 'decision_trees')
# if not results_path.exists():
#     results_path.mkdir(parents=True)
pd.core.common.is_list_like = pd.api.types.is_list_like

##### Unused packages #####
# import yfinance as yf
# yf.pdr_override()
# from talib import RSI, BBANDS, MACD, NATR, ATR
# from pandas_datareader import data as pdr
# import pyfolio as pf
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
# from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, make_scorer
# import graphviz
# import matplotlib.pyplot as plt
# import statsmodels
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import coint, adfuller


def get_table(title, filename, match, use_cache=False):
    """
    Fetch table from Wikipedia and save to CSV.
    
    Args:
        title: Wikipedia page title
        filename: CSV file to save/load
        match: Pattern to match in HTML tables
        use_cache: Whether to use cached CSV if exists
    """
    if not (use_cache and os.path.isfile(filename)):
        html = wp.page(title).html()
        df = pd.read_html(io.StringIO(html), header=0, match=match)[0]
        df.to_csv(filename, header=True, index=False, encoding='utf-8')
    
    return pd.read_csv(filename)


# Fetch S&P 500 company list from Wikipedia
title = 'List of S&P 500 companies'
filename = 'sp500.csv'
df = get_table(title, filename, match='Symbol')

# dd/mm/YY H:M:S
now = datetime.datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print('{} (retrieved {})'.format(title, dt_string))

# Extract ticker symbols from the dataframe
tickers = df['Symbol'].to_list()

# Define the date range for historical data
start = '2010-01-01'
end = pd.to_datetime('today').strftime('%Y-%m-%d')

from yahooquery import Ticker

# ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List of S&P 500 companies')
# df = ticker_list[0]
# tickers = df['Symbol'].to_list()

# Use asynchronous=True for faster data fetching
tickers_list = Ticker(tickers, asynchronous=True)
daily = tickers_list.history(interval='1d', start=start, end=end)

# Pivoting the DataFrame to match the desired output format
pivot_df = daily.reset_index().pivot(index='date', columns='symbol', values=['close', 'adjclose', 'volume','open','low','high'])
pivot_df.rename(columns={
    'close': 'Close',
    'adjclose': 'Adj Close',
    'volume': 'Volume',
    'open': 'Open',
    'low': 'Low',
    'high': 'High'
}, level=0, inplace=True)
pivot_df.columns.names = ['Metric', 'Ticker']
pivot_df.reset_index(inplace=True)
pivot_df.rename(columns={'date': 'Date'}, inplace=True)
daily = pivot_df.set_index('Date')



# dict1 = pd.read_excel("/content/drive/MyDrive/Colab Notebooks/Quant/industry.xlsx", sheet_name = ['SP500'])
# Load industry/sector information from Excel file
dict1 = pd.read_excel("G:/My Drive/Colab Notebooks/Quant/industry.xlsx", sheet_name = ['STI','SP500','SP500_rating'])

# Optimize sector lookup using set for faster membership testing
sp500_tickers_set = set(dict1['SP500']['tickers'].to_list())
tickers_missing = [t for t in tickers if t not in sp500_tickers_set]

sector = []
drop_list = list(sp500_tickers_set)

# Batch fetch sector info for missing tickers (much faster than individual calls)
if tickers_missing:
    tickers_sector = yq.Ticker(tickers_missing, asynchronous=True)
    profiles = tickers_sector.asset_profile
    
    for ticker in tickers_missing:
        try:
            sector_value = profiles[ticker]['sector']
            print(ticker, sector_value)
            sector.append(sector_value)
        except:
            print(ticker)
            drop_list.append(ticker)

# Create a set for faster lookup and filter tickers
drop_set = set(drop_list)
tickers = [tick for tick in tickers if tick not in drop_set]

# Combine manually fetched sector data with industry file data
# industry = pd.DataFrame({'tickers':tickers,'sector': sector})
industry = pd.concat([pd.DataFrame({'tickers':tickers,'sector': sector}),dict1['SP500']], axis = 0)

# Optimize: Combine melt operations more efficiently
# melted_dfs = []
# for metric, col_name in [('Adj Close', 'adj close'), ('Open', 'open'), ('High', 'high'), 
#                           ('Low', 'low'), ('Close', 'close'), ('Volume', 'volume')]:
#     melted = pd.melt(daily[metric], ignore_index=False, var_name='ticker', value_name=col_name)
#     melted_dfs.append(melted)

# Combine all melted dataframes into a single price dataframe
# price = pd.concat(melted_dfs, axis=1, join='outer')
# price = price.loc[:, ~price.columns.duplicated()]
# prices = price.reset_index().rename(columns={'Date': 'date'}).set_index(['ticker', 'date'])

# Strategy parameters
x = 3  # Lookback period (months) - used to calculate momentum
n = 3  # Number of top performing stocks to select
d = 0  # Days offset for return calculation
# h = 1  # Holding period (months) - currently unused
# m = 0  # Number of top skips - currently unused
# s = 0.85  # Stop loss threshold for the stock - currently unused
l = 1.07  # Limit/take profit threshold for the stock
# To exclude multiple different sectors

# Calculate daily returns from adjusted close prices
daily_ret = daily['Adj Close'].pct_change(fill_method=None)
daily_ret.index=pd.to_datetime(daily_ret.index)+ pd.DateOffset(days=d)

# Calculate monthly returns by compounding daily returns within each month
monthly_ret = (daily_ret+1).groupby(pd.Grouper(freq="ME")).prod()

# Calculate cumulative returns from start of each month
ret_calc = (daily_ret+1).groupby(pd.Grouper(freq="ME")).cumprod()

# Generate rolling returns for x months and drop the first x-1 months
rolling_ret = monthly_ret.rolling(x, min_periods=x).agg(lambda x: x.prod()).dropna(axis=0)


# def portfolio_variance(weights, cov_matrix):
#     """Calculate portfolio variance given weights and covariance matrix."""
#     return np.dot(weights.T, np.dot(cov_matrix, weights))


def build_drop_list(daily_ret, monthly_ret, i, x):
    """
    Build drop list based on filtering criteria.
    Filters out stocks with undesirable return characteristics.
    
    Args:
        daily_ret: Daily returns dataframe
        monthly_ret: Monthly returns dataframe
        i: Current month index
        x: Lookback period in months
    
    Returns:
        List of tickers to exclude from portfolio
    """
    # Cache date calculations
    month_idx_3 = monthly_ret.index[x + i - 3]
    month_idx_2 = monthly_ret.index[x + i - 2]
    month_idx_1 = monthly_ret.index[x + i - 1]
    
    # Calculate date boundaries for filtering
    start_3 = pd.offsets.MonthBegin().rollback(month_idx_3)
    end_2 = pd.offsets.MonthEnd().rollforward(month_idx_2)
    start_1 = pd.offsets.MonthBegin().rollback(month_idx_1)
    end_1 = pd.offsets.MonthEnd().rollforward(month_idx_1)
    
    # Cache filtered dataframes
    period_3_2 = daily_ret[(daily_ret.index >= start_3) & (daily_ret.index <= end_2)]
    period_1 = daily_ret[(daily_ret.index >= start_1) & (daily_ret.index <= end_1)]
    
    drop_list = []
    
    # Exclude stocks with >10% return in a single day (potential data issues)
    drop_list.extend(period_3_2.columns[period_3_2.max() >= 0.1].tolist())
    
    # Keep stocks with sharp decline of 5% return in a day (avoid extreme losers)
    drop_list.extend(period_3_2.columns[~(period_3_2.min() <= -0.05)].tolist())
    
    # Keep stocks with average daily return >= 0.01 (positive momentum)
    drop_list.extend(period_1.columns[period_1.mean() >= 0.01].tolist())
    
    return list(set(drop_list))  # Remove duplicates


# Pre-compute industry lookup for efficiency
industry_clean = industry.dropna(subset='tickers').drop_duplicates(subset='tickers').set_index('tickers')
industry_indexed = industry.set_index('tickers')

# Main momentum strategy loop
ret = pd.DataFrame()
for i in range(len(monthly_ret) - x):
    # Apply filtering criteria to exclude stocks
    drop_list = build_drop_list(daily_ret, monthly_ret, i, x)
    
    # Filter rolling returns by dropping excluded stocks
    rolling_ret_filtered = rolling_ret.drop(drop_list, axis=1)
    rolling_ret_row = rolling_ret_filtered.iloc[i]
    
    # Obtain the top n stocks with highest return by sector
    ret_sector = pd.concat([
        rolling_ret_row.to_frame('return'),
        industry_clean
    ], axis=1, join='inner').reset_index()
    
    # Select top performing stock from each sector
    top_n_industry = (ret_sector.loc[ret_sector.groupby('sector')['return'].transform('max') == ret_sector['return']]
                      .sort_values(by='return', ascending=False)
                      .iloc[:3]
                      .set_index('index'))
    top_n = top_n_industry
    
    # print("check return on:", rolling_ret.iloc[i].name)
    month_idx_3 = monthly_ret.index[x + i - 3] 
    # month_idx_1 = monthly_ret.index[x + i - 1] # unused
    # print('Check return period ', pd.offsets.MonthBegin().rollback(month_idx_3), 
        #   pd.offsets.MonthEnd().rollforward(month_idx_1))

    # Calculate date boundaries for covariance calculation
    start_1 = pd.offsets.MonthBegin().rollback(month_idx_3)
    end_1 = pd.offsets.MonthEnd().rollforward(month_idx_3)
    period_1_filtered = daily_ret[(daily_ret.index >= start_1) & (daily_ret.index <= end_1)][top_n.index]
    # covariance_matrix = period_1_filtered.cov() * 252  # Annualized covariance - currently unused

    # Calculate optimal weights (currently equal weights)
    optimal_weights = np.ones(len(top_n)) / len(top_n)
    
    # Calculate cumulative returns for the holding period
    next_month = rolling_ret.iloc[i + 1].name + pd.offsets.MonthBegin(-1)
    date_filter = (ret_calc.index + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1) == next_month)
    
    if len(top_n) != n:
        combined = pd.Series(1, index=ret_calc[date_filter].columns)
    else:
        cum_df = ret_calc[date_filter][top_n.index].copy()
        
        # Apply take profit limit: exit position if price rises above limit threshold
        for j in top_n.index:
            try:
                high_data = daily['High'][j][date_filter]
                open_price = daily['Open'][j].shift(1)[date_filter].iloc[0]
                condition = (high_data / open_price > l)
                if condition.any():
                    trigger_date = condition[condition].index[0]
                    trigger_idx = cum_df.index.get_loc(trigger_date)
                    cum_df[j].iloc[trigger_idx:] = l
            except (IndexError, KeyError):
                pass
                # print('Limit sell not triggered')
        
        # Multiply each column by its weight, then sum across columns
        # Align weights with columns using Series for proper broadcasting
        weights_series = pd.Series(optimal_weights, index=top_n.index)
        combined = (cum_df * weights_series).sum(axis=1)
    
    # Only process recent months (last 3 months)
    if next_month >= pd.to_datetime('today') + pd.offsets.MonthBegin(-3):
        # Get the 3-month lookback period dates
        lookback_start = rolling_ret.iloc[i].name - pd.DateOffset(months=x-1)
        lookback_end = rolling_ret.iloc[i].name
        
        print(f"\n=== Month: {next_month.strftime('%Y-%m')} ===")
        print(f"Lookback period: {lookback_start.strftime('%Y-%m')} to {lookback_end.strftime('%Y-%m')}")
        print(f"Selected stocks (past 3-month return, sector):")
        
        for ticker in top_n.index:
            past_return = rolling_ret_row[ticker]
            sector_val = industry_indexed.loc[ticker, 'sector'] if ticker in industry_indexed.index else 'N/A'
            sector = sector_val.values[0] if hasattr(sector_val, 'values') else sector_val
            print(f"  {ticker}: {past_return*100:.2f}% ({sector})")
        
        # Get current month cumulative returns
        if len(top_n) == n:
            print(f"Current month cumulative return:")
            for ticker in top_n.index:
                final_return = cum_df[ticker].iloc[-1] - 1
                print(f"  {ticker}: {final_return*100:.2f}%")
            portfolio_return = (combined.iloc[-1] / combined.iloc[0] - 1) if len(combined) > 0 else 0
            print(f"Portfolio return (equal-weighted): {portfolio_return*100:.2f}%")
        print("-" * 50)
        
        # Obtain average daily return of portfolio
        combined_ret = combined.div(combined.shift(1)).fillna(combined.iloc[0])
        if not combined_ret.empty:
            ret = pd.concat([ret, combined_ret], axis=0)

# Set timezone for return index
ret.index = (ret.index - pd.DateOffset(days=d)).tz_localize('UTC')

### Next Month's 3 Company - Predict next month's top momentum stocks
# Use the same optimized function instead of duplicating code
i = len(monthly_ret) - x  # Continue from where loop ended
drop_list = build_drop_list(daily_ret, monthly_ret, i, x)

# Obtain the top n stocks with highest return
rolling_ret_filtered = rolling_ret.drop(drop_list, axis=1)
rolling_ret_row = rolling_ret_filtered.iloc[i]

ret_sector = pd.concat([
    rolling_ret_row.to_frame('return'),
    industry_clean
], axis=1, join='inner').reset_index()

top_n_industry = (ret_sector.loc[ret_sector.groupby('sector')['return'].transform('max') == ret_sector['return']]
                  .sort_values(by='return', ascending=False)
                  .iloc[:3]
                  .set_index('index'))
top_n = top_n_industry

# Print prediction details
lookback_start = rolling_ret.iloc[i].name - pd.DateOffset(months=x-1)
lookback_end = rolling_ret.iloc[i].name

# Check if next month data is available for prediction
if i + 1 < len(rolling_ret):
    next_month = rolling_ret.iloc[i + 1].name + pd.offsets.MonthBegin(-1)
    has_next_month_data = True
else:
    next_month = rolling_ret.iloc[i].name + pd.DateOffset(months=1)
    has_next_month_data = False

print(f"\n=== PREDICTION for {next_month.strftime('%Y-%m')} ===")
if not has_next_month_data:
    print("(Next month data not available - this is a forward prediction)")
print(f"Lookback period: {lookback_start.strftime('%Y-%m')} to {lookback_end.strftime('%Y-%m')}")
print(f"Selected stocks (past 3-month return, sector):")
for ticker in top_n.index:
    past_return = rolling_ret_row[ticker]
    sector_val = industry_indexed.loc[ticker, 'sector'] if ticker in industry_indexed.index else 'N/A'
    sector = sector_val.values[0] if hasattr(sector_val, 'values') else sector_val
    print(f"  {ticker}: {past_return*100:.2f}% ({sector})")
print("-" * 50)
