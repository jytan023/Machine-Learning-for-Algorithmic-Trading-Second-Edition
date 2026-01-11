import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

import numpy as np
import pandas as pd
# from scipy.optimize import minimize
import os
# import seaborn as sns; sns.set(style="whitegrid")
import datetime
import yahooquery as yq
from pathlib import Path
import wikipedia as wp
import io

results_path = Path('results', 'decision_trees')
if not results_path.exists():
    results_path.mkdir(parents=True)
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
    if not (use_cache and os.path.isfile(filename)):
        html = wp.page(title).html()
        df = pd.read_html(io.StringIO(html), header=0, match=match)[0]
        df.to_csv(filename, header=True, index=False, encoding='utf-8')
    
    return pd.read_csv(filename)

title = 'List of S&P 500 companies'
filename = 'sp500.csv'
df = get_table(title, filename, match='Symbol')

# dd/mm/YY H:M:S
now = datetime.datetime.now()
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print('{} (retrieved {})'.format(title, dt_string))

# df = ticker_list[0]
tickers = df['Symbol'].to_list()

start = '2010-01-01'
end = pd.to_datetime('today').strftime('%Y-%m-%d')

from yahooquery import Ticker

# ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# df = ticker_list[0]
# tickers = df['Symbol'].to_list()

tickers_list = Ticker(tickers, asynchronous=False)

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
dict1 = pd.read_excel("G:/My Drive/Colab Notebooks/Quant/industry.xlsx", sheet_name = ['STI','SP500','SP500_rating'])

# Optimize sector lookup using set for faster membership testing
sp500_tickers_set = set(dict1['SP500']['tickers'].to_list())
drop_list = list(sp500_tickers_set)
sector = []

for ticker in tickers:
    if ticker not in sp500_tickers_set:
        item = yq.Ticker(ticker)
        try:
            sector_value = item.asset_profile[list(item.asset_profile)[0]]['sector']
            print(ticker, sector_value)
            sector.append(sector_value)
        except:
            print(ticker)
            drop_list.append(ticker)

drop_set = set(drop_list)
tickers = [tick for tick in tickers if tick not in drop_set]

# industry = pd.DataFrame({'tickers':tickers,'sector': sector})
industry = pd.concat([pd.DataFrame({'tickers':tickers,'sector': sector}),dict1['SP500']], axis = 0)

# Optimize: Combine melt operations more efficiently
melted_dfs = []
for metric, col_name in [('Adj Close', 'adj close'), ('Open', 'open'), ('High', 'high'), 
                          ('Low', 'low'), ('Close', 'close'), ('Volume', 'volume')]:
    melted = pd.melt(daily[metric], ignore_index=False, var_name='ticker', value_name=col_name)
    melted_dfs.append(melted)

price = pd.concat(melted_dfs, axis=1, join='outer')
price = price.loc[:, ~price.columns.duplicated()]
prices = price.reset_index().rename(columns={'Date': 'date'}).set_index(['ticker', 'date'])

x = 3 # Lookback period (months)
n = 3 # Number of top performing stocks
d = 0 # Days offset
h = 1 # Holding period (months)
m = 0 # Number of top skips
s = 0.85 # stop loss for the stock
l = 1.07 # limit close for the stock
# To exclude multiple different sectors

# Obtain daily return of stocks
daily_ret = daily['Adj Close'].pct_change()
daily_ret.index=pd.to_datetime(daily_ret.index)+ pd.DateOffset(days=d)

# Obtain monthly return of stocks
monthly_ret = (daily_ret+1).groupby(pd.Grouper(freq="M")).prod()

# Obtain cumulated return of stocks from start of month
ret_calc = (daily_ret+1).groupby(pd.Grouper(freq="M")).cumprod()

# Generate rolling returns for x months and drop the first x-1 months
rolling_ret = monthly_ret.rolling(x, min_periods=x).agg(lambda x: x.prod()).dropna(axis=0)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def build_drop_list(daily_ret, monthly_ret, i, x):
    """
    Build drop list based on filtering criteria.
    Optimized to cache date calculations and reduce repeated operations.
    """
    # Cache date calculations
    month_idx_3 = monthly_ret.index[x + i - 3]
    month_idx_2 = monthly_ret.index[x + i - 2]
    month_idx_1 = monthly_ret.index[x + i - 1]
    
    start_3 = pd.offsets.MonthBegin().rollback(month_idx_3)
    end_2 = pd.offsets.MonthEnd().rollforward(month_idx_2)
    start_1 = pd.offsets.MonthBegin().rollback(month_idx_1)
    end_1 = pd.offsets.MonthEnd().rollforward(month_idx_1)
    
    # Cache filtered dataframes
    period_3_2 = daily_ret[(daily_ret.index >= start_3) & (daily_ret.index <= end_2)]
    period_1 = daily_ret[(daily_ret.index >= start_1) & (daily_ret.index <= end_1)]
    
    drop_list = []
    
    # Exclude stocks with >10% return in a day
    drop_list.extend(period_3_2.columns[period_3_2.max() >= 0.1].tolist())
    
    # Keep stocks with sharp decline of 5% return in a day
    drop_list.extend(period_3_2.columns[~(period_3_2.min() <= -0.05)].tolist())
    
    # Keep stocks with average > 0.01 return
    drop_list.extend(period_1.columns[period_1.mean() >= 0.01].tolist())
    
    return list(set(drop_list))  # Remove duplicates

# Pre-compute industry lookup for efficiency
industry_clean = industry.dropna(subset='tickers').drop_duplicates(subset='tickers').set_index('tickers')
industry_indexed = industry.set_index('tickers')

ret = pd.DataFrame()
for i in range(len(monthly_ret) - x):
    # Use optimized drop_list function
    drop_list = build_drop_list(daily_ret, monthly_ret, i, x)
    
    # Cache filtered rolling_ret to avoid repeated operations
    rolling_ret_filtered = rolling_ret.drop(drop_list, axis=1)
    rolling_ret_row = rolling_ret_filtered.iloc[i]
    
    # Obtain the top n stocks with highest return
    ret_sector = pd.concat([
        rolling_ret_row.to_frame('return'),
        industry_clean
    ], axis=1, join='inner').reset_index()
    
    top_n_industry = (ret_sector.loc[ret_sector.groupby('sector')['return'].transform('max') == ret_sector['return']]
                      .sort_values(by='return', ascending=False)
                      .iloc[:3]
                      .set_index('index'))
    top_n = top_n_industry
    
    print("check return on:", rolling_ret.iloc[i].name)
    month_idx_3 = monthly_ret.index[x + i - 3]
    month_idx_1 = monthly_ret.index[x + i - 1]
    print('Check return period ', pd.offsets.MonthBegin().rollback(month_idx_3), 
          pd.offsets.MonthEnd().rollforward(month_idx_1))

    # Cache date calculations for covariance
    start_1 = pd.offsets.MonthBegin().rollback(month_idx_1)
    end_1 = pd.offsets.MonthEnd().rollforward(month_idx_1)
    period_1_filtered = daily_ret[(daily_ret.index >= start_1) & (daily_ret.index <= end_1)][top_n.index]
    covariance_matrix = period_1_filtered.cov() * 252

    # Calculate optimal weights (currently equal weights)
    optimal_weights = np.ones(len(top_n)) / len(top_n)
    
    # Calculate cumulative returns
    next_month = rolling_ret.iloc[i + 1].name + pd.offsets.MonthBegin(-1)
    date_filter = (ret_calc.index + pd.offsets.MonthEnd(0) - pd.offsets.MonthBegin(1) == next_month)
    
    if len(top_n) != n:
        combined = pd.Series(1, index=ret_calc[date_filter].columns)
    else:
        cum_df = ret_calc[date_filter][top_n.index].copy()
        
        # Cache date filter for daily data
        for j in top_n.index:
            # Limit take profit
            try:
                high_data = daily['High'][j][date_filter]
                open_price = daily['Open'][j].shift(1)[date_filter].iloc[0]
                condition = (high_data / open_price > l)
                if condition.any():
                    trigger_date = condition[condition].index[0]
                    trigger_idx = cum_df.index.get_loc(trigger_date)
                    cum_df[j].iloc[trigger_idx:] = l
            except (IndexError, KeyError):
                print('Limit sell not triggered')
        
        # Multiply each column by its weight, then sum across columns
        # Align weights with columns using Series for proper broadcasting
        weights_series = pd.Series(optimal_weights, index=top_n.index)
        combined = (cum_df * weights_series).sum(axis=1)
    
    # Only process recent months
    if next_month >= pd.to_datetime('today') + pd.offsets.MonthBegin(-3):
        print("following month's return", next_month)
        print(top_n, optimal_weights, industry_indexed.loc[top_n.index, :])
        
        # Obtain average daily return of portfolio
        ret = pd.concat([ret, combined.div(combined.shift(1)).fillna(combined.iloc[0])], axis=0)

ret.index = (ret.index - pd.DateOffset(days=d)).tz_localize('UTC')

### Next Month's 3 Company
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
print(top_n, industry_indexed.loc[top_n.index, :])