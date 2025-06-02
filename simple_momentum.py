import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")

import pyfolio as pf

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, make_scorer
import graphviz

import statsmodels.api as sm

pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
# yf.pdr_override()
# from talib import RSI, BBANDS, MACD, NATR, ATR
import yahooquery as yq

from pathlib import Path
results_path = Path('results', 'decision_trees')
if not results_path.exists():
    results_path.mkdir(parents=True)

start = '2010-01-01'
end = pd.to_datetime('today').strftime('%Y-%m-%d')

from yahooquery import Ticker

ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = ticker_list[0]
tickers = df['Symbol'].to_list()

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

sector = []
drop_list = []
drop_list = dict1['SP500']['tickers'].to_list()
for ticker in tickers:
  if ticker not in dict1['SP500']['tickers'].to_list():
    item = yq.Ticker(ticker)
    try:
      print(ticker,item.asset_profile[list(item.asset_profile)[0]]['sector'])
      sector.append(item.asset_profile[list(item.asset_profile)[0]]['sector'])
    except:
      print(ticker)
      drop_list.append(ticker)
      # continue
tickers = [tick for tick in tickers if tick not in drop_list]

# industry = pd.DataFrame({'tickers':tickers,'sector': sector})
industry = pd.concat([pd.DataFrame({'tickers':tickers,'sector': sector}),dict1['SP500']], axis = 0)

aclose = pd.melt(daily['Adj Close'], ignore_index=False, var_name='ticker',value_name='adj close')
open = pd.melt(daily['Open'], ignore_index=False, var_name='ticker',value_name='open')
high = pd.melt(daily['High'], ignore_index=False, var_name='ticker',value_name='high')
low = pd.melt(daily['Low'], ignore_index=False, var_name='ticker',value_name='low')
close = pd.melt(daily['Close'], ignore_index=False, var_name='ticker',value_name='close')
volume = pd.melt(daily['Volume'], ignore_index=False, var_name='ticker',value_name='volume')
price = pd.concat([aclose,open,high,low,close,volume], axis=1, join='outer')
price = price.loc[:,~price.columns.duplicated()]
prices = price.reset_index().rename(columns = {'Date':'date'}).set_index(['ticker','date'])

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
rolling_ret = monthly_ret.rolling(x, min_periods=x).agg(lambda x : x.prod()).dropna(axis=0)

def portfolio_variance(weights, cov_matrix):
      return np.dot(weights.T, np.dot(cov_matrix, weights))

ret = pd.DataFrame()
for i in range(len(monthly_ret)-x):
  drop_list = []
  # # exclude list of stock with more than 10% return in a day within the period (only 2 months??)
  drop_list = daily_ret.transpose().index[(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-3])) &
                                                     (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-2]))].max()>=0.1)].tolist()

  # # keep list of stock with sharp decline of 5% return in a day within the period (only 2 months??)
  drop_list = drop_list + daily_ret.transpose().index[~(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-3])) &
                                                                 (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-2]))].min()<=-0.05)].tolist()

  # # Keep list of stocks that have average > 0.01 return
  drop_list = drop_list + daily_ret.transpose().index[(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-1])) &
                                                                 (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-1]))].mean() >= 0.01)].tolist()
  # # Keep list of stocks that have below > 0.1 variance
  # drop_list = drop_list + daily_ret.transpose().index[(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-1])) &
  #                                                                (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-1]))].var()* 252 >= 0.08)].tolist()

  # Obtain the top n stocks with highest return
  top_n = rolling_ret.drop(drop_list, axis=1).iloc[i].transpose().nlargest(n)
  ret_sector = pd.concat([rolling_ret.drop(drop_list, axis=1).iloc[i].transpose(),industry.dropna(subset = 'tickers').drop_duplicates(subset='tickers').set_index('tickers')],axis=1).reset_index().rename({rolling_ret.drop(drop_list, axis=1).iloc[i].transpose().name:'return'}, axis=1)
  top_n_industry = ret_sector.loc[ret_sector.groupby('sector')['return'].transform('max') == ret_sector['return']].sort_values(by = 'return', ascending=False).iloc[:3].set_index('index')
  top_n = top_n_industry
  print("check return on:", rolling_ret.iloc[i].name)
  print('Check return period ', pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-3]), pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-1]))

  # Obtain the average cumulative return since start of month of the top n stocks
  # Obtain weight for top 5 stock
  expected_returns =  rolling_ret.iloc[i].transpose().nlargest(n) * 12 / x
  initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

  # expected_returns =  monthly_ret.iloc[x+i-1][top_n.index] * 12

  ### previously using all time variance
  # covariance_matrix = daily_ret[daily_ret.index +  pd.offsets.MonthBegin(-1) >= rolling_ret.iloc[i].name + pd.offsets.MonthBegin(-1)][top_n.index].cov() * 252

  ### Now use past 1 month variance
  covariance_matrix = daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-1])) &
                                                                 (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-1]))][top_n.index].cov() * 252

  # constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
  initial_weights = np.ones(len(expected_returns)) / len(expected_returns)
  # bounds = [(0, 1) for _ in range(len(expected_returns))]
  # result = minimize(portfolio_variance, initial_weights, args=(covariance_matrix,),
  #                   method='SLSQP', constraints=constraints, bounds=bounds)
  # optimal_weights = result.x
  optimal_weights = initial_weights
  if len(top_n) !=n:
    combined = pd.Series(1, index = (ret_calc[(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))]).transpose().sum().index)
  else:
    cum_df = ret_calc[(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))][top_n.index]
    for j in top_n.index:
      # Limit take profit
      try:
        # cum_df[j].iloc[cum_df.index.get_loc(cum_df[cum_df[j] > l].index[0]):] = cum_df[j].iloc[cum_df.index.get_loc(cum_df[cum_df[j] > l].index[0])]
        high = daily['High'][j][(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))]
        open = daily['Open'][j].shift(1)[(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))].iloc[0]
        cum_df[j].iloc[cum_df.index.get_loc(cum_df[high/open >l].index[0]):] = l
      except:
        print('Limit sell not triggered')
      # Stop loss
      # try:
      #   # cum_df[j].iloc[cum_df.index.get_loc(cum_df[cum_df[j] < s].index[0]):] = cum_df[j].iloc[cum_df.index.get_loc(cum_df[cum_df[j] < s].index[0])]
      #   low = daily['Low'][j][(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))]
      #   open = daily['Open'][j].shift(1)[(ret_calc.index + pd.offsets.MonthEnd(0) -  pd.offsets.MonthBegin(1) == rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))].iloc[0]
      #   cum_df[j].iloc[cum_df.index.get_loc(cum_df[low/open < s].index[0]):] = s
      # except:
      #   print('Stop loss not triggered')
    combined = (cum_df*optimal_weights).transpose().sum()
  if  rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1) >= pd.to_datetime('today') + pd.offsets.MonthBegin(-3):
    print("following month's return", rolling_ret.iloc[i+1].name + pd.offsets.MonthBegin(-1))
    print(top_n, optimal_weights, industry.set_index('tickers').loc[top_n.index,:])

    # Obtain average daily return of portfolio by dividing the average of r_t+1/r_t
    ret = pd.concat([ret,combined.div(combined.shift(1)).fillna(combined.iloc[0])], axis=0)
    # print(top_n, optimal_weights, industry.set_index('tickers').loc[top_n.index,:])
ret.index = (ret.index-pd.DateOffset(days=d)).tz_convert('utc')


### Next Month's 3 Company
i+=1
drop_list = daily_ret.transpose().index[(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-3])) &
                                                    (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-2]))].max()>=0.1)].tolist()

# keep list of stock with sharp decline of 5% return in a day within the period
drop_list = drop_list + daily_ret.transpose().index[~(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-3])) &
                                                                (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-2]))].min()<=-0.05)].tolist()

# Keep list of stocks that have average > 0 return
drop_list = drop_list + daily_ret.transpose().index[(daily_ret[(daily_ret.index >= pd.offsets.MonthBegin().rollback(monthly_ret.index[x+i-1])) &
                                                                (daily_ret.index <= pd.offsets.MonthEnd().rollforward(monthly_ret.index[x+i-1]))].mean() >= 0.01)].tolist()
# Obtain the top n stocks with highest return
top_n = rolling_ret.drop(drop_list, axis=1).iloc[i].transpose().nlargest(n)
ret_sector = pd.concat([rolling_ret.drop(drop_list, axis=1).iloc[i].transpose(),industry.dropna(subset = 'tickers').drop_duplicates(subset='tickers').set_index('tickers')],axis=1).reset_index().rename({rolling_ret.drop(drop_list, axis=1).iloc[i].transpose().name:'return'}, axis=1)
top_n_industry = ret_sector.loc[ret_sector.groupby('sector')['return'].transform('max') == ret_sector['return']].sort_values(by = 'return', ascending=False).iloc[:3].set_index('index')
top_n = top_n_industry
print(top_n,  industry.set_index('tickers').loc[top_n.index,:])