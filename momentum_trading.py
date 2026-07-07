"""
Momentum Trading Strategy with Webull Trading (Official SDK)
=============================================================
Runs the momentum strategy and executes trades based on predictions.
"""

import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

import numpy as np
import pandas as pd
import os
import datetime
import yahooquery as yq
from pathlib import Path
import wikipedia as wp
import io
from dotenv import load_dotenv

load_dotenv()

pd.core.common.is_list_like = pd.api.types.is_list_like


WEBULL_EMAIL = os.getenv("WEBULL_EMAIL")
WEBULL_PASSWORD = os.getenv("WEBULL_PASSWORD")
WEBULL_DEVICE_ID = os.getenv("WEBULL_DEVICE_ID")


class MomentumTrader:
    def __init__(self, dry_run=True, paper=True):
        """
        Initialize the momentum trader.
        
        Args:
            dry_run: If True, only simulate trades without executing
            paper: If True, use paper trading. If False, use live trading.
        """
        self.dry_run = dry_run
        self.paper = paper
        self.trader = None
        
        if not dry_run:
            self._init_webull()
    
    def _init_webull(self):
        """Initialize Webull connection."""
        from webull import webull
        from webull.trade import webull_trade
        
        wb = webull()
        if self.paper:
            self.trader = webull_trade(wb, paper_account=True)
            print("Connected to Webull Paper Trading")
        else:
            self.trader = webull_trade(wb, paper_account=False)
            print("Connected to Webull Live Trading")
        
        wb.login(WEBULL_EMAIL, WEBULL_PASSWORD, WEBULL_DEVICE_ID)
        self.trader.get_account_id()
        print("Logged in to Webull")
    
    def get_account_info(self):
        """Get account information."""
        if self.trader:
            account = self.trader.get_account()
            return {
                'cash': float(account.get('cashBalance', 0)),
                'buying_power': float(account.get('buyingPower', 0))
            }
        return {'cash': 0, 'buying_power': 0}
    
    def get_current_positions(self):
        """Get current stock positions."""
        if self.trader:
            positions = self.trader.get_positions()
            return {p.get('symbol'): int(p.get('quantity', 0)) for p in positions}
        return {}
    
    def get_stock_price(self, ticker):
        """Get current stock price from Webull."""
        if self.trader:
            from webull import webull
            wb = webull()
            wb.login(WEBULL_EMAIL, WEBULL_PASSWORD, WEBULL_DEVICE_ID)
            quote = wb.get_quote(ticker)
            return float(quote.get('close', 0))
        return None
    
    def place_buy_order(self, ticker, quantity):
        """Place a buy order."""
        if self.dry_run:
            print(f"[DRY RUN] BUY {quantity} shares of {ticker}")
            return {"orderId": "dry_run"}
        
        order = self.trader.place_order(
            stock=ticker,
            action="BUY",
            orderType="MKT",
            quantity=quantity,
            timeInForce="GTC"
        )
        print(f"BUY order placed: {quantity} shares of {ticker}")
        return order
    
    def place_sell_order(self, ticker, quantity):
        """Place a sell order."""
        if self.dry_run:
            print(f"[DRY RUN] SELL {quantity} shares of {ticker}")
            return {"orderId": "dry_run"}
        
        order = self.trader.place_order(
            stock=ticker,
            action="SELL",
            orderType="MKT",
            quantity=quantity,
            timeInForce="GTC"
        )
        print(f"SELL order placed: {quantity} shares of {ticker}")
        return order


def get_table(title, filename, match, use_cache=False):
    """Fetch table from Wikipedia and save to CSV."""
    if not (use_cache and os.path.isfile(filename)):
        html = wp.page(title).html()
        df = pd.read_html(io.StringIO(html), header=0, match=match)[0]
        df.to_csv(filename, header=True, index=False, encoding='utf-8')
    return pd.read_csv(filename)


def run_momentum_strategy():
    """Run the momentum strategy and return predictions."""
    
    title = 'List of S&P 500 companies'
    filename = 'sp500.csv'
    df = get_table(title, filename, match='Symbol')
    
    tickers = df['Symbol'].to_list()
    start = '2010-01-01'
    end = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    tickers_list = yq.Ticker(tickers, asynchronous=True)
    daily = tickers_list.history(interval='1d', start=start, end=end)
    
    pivot_df = daily.reset_index().pivot(index='date', columns='symbol', values=['close', 'adjclose', 'volume','open','low','high'])
    pivot_df.rename(columns={
        'close': 'Close', 'adjclose': 'Adj Close', 'volume': 'Volume',
        'open': 'Open', 'low': 'Low', 'high': 'High'
    }, level=0, inplace=True)
    pivot_df.columns.names = ['Metric', 'Ticker']
    pivot_df.reset_index(inplace=True)
    pivot_df.rename(columns={'date': 'Date'}, inplace=True)
    daily = pivot_df.set_index('Date')
    
    dict1 = pd.read_excel("G:/My Drive/Colab Notebooks/Quant/industry.xlsx", sheet_name = ['STI','SP500','SP500_rating'])
    
    sp500_tickers_set = set(dict1['SP500']['tickers'].to_list())
    tickers_missing = [t for t in tickers if t not in sp500_tickers_set]
    
    sector = []
    drop_list = list(sp500_tickers_set)
    
    if tickers_missing:
        tickers_sector = yq.Ticker(tickers_missing, asynchronous=True)
        profiles = tickers_sector.asset_profile
        
        for ticker in tickers_missing:
            try:
                sector_value = profiles[ticker]['sector']
                sector.append(sector_value)
            except:
                drop_list.append(ticker)
    
    drop_set = set(drop_list)
    tickers = [tick for tick in tickers if tick not in drop_set]
    
    industry = pd.concat([pd.DataFrame({'tickers':tickers,'sector': sector}),dict1['SP500']], axis = 0)
    
    x = 3
    n = 3
    d = 0
    l = 1.07
    
    daily_ret = daily['Adj Close'].pct_change(fill_method=None)
    daily_ret.index = pd.to_datetime(daily_ret.index) + pd.DateOffset(days=d)
    
    monthly_ret = (daily_ret+1).groupby(pd.Grouper(freq="ME")).prod()
    ret_calc = (daily_ret+1).groupby(pd.Grouper(freq="ME")).cumprod()
    
    rolling_ret = monthly_ret.rolling(x, min_periods=x).agg(lambda x: x.prod()).dropna(axis=0)
    
    industry_clean = industry.dropna(subset='tickers').drop_duplicates(subset='tickers').set_index('tickers')
    industry_indexed = industry.set_index('tickers')
    
    i = len(monthly_ret) - x
    
    month_idx_3 = monthly_ret.index[x + i - 3]
    
    start_3 = pd.offsets.MonthBegin().rollback(month_idx_3)
    end_2 = pd.offsets.MonthEnd().rollforward(month_idx_3)
    
    period_3_2 = daily_ret[(daily_ret.index >= start_3) & (daily_ret.index <= end_2)]
    
    drop_list = []
    drop_list.extend(period_3_2.columns[period_3_2.max() >= 0.1].tolist())
    drop_list.extend(period_3_2.columns[~(period_3_2.min() <= -0.05)].tolist())
    
    start_1 = pd.offsets.MonthBegin().rollback(month_idx_3)
    end_1 = pd.offsets.MonthEnd().rollforward(month_idx_3)
    period_1 = daily_ret[(daily_ret.index >= start_1) & (daily_ret.index <= end_1)]
    drop_list.extend(period_1.columns[period_1.mean() >= 0.01].tolist())
    drop_list = list(set(drop_list))
    
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
    
    return top_n, rolling_ret_row, industry_indexed, daily


def execute_trades(predicted_stocks, trader, portfolio_value=10000):
    """
    Execute trades based on predicted stocks.
    
    Args:
        predicted_stocks: List of ticker symbols to buy
        trader: MomentumTrader instance
        portfolio_value: Total value to allocate
    """
    allocation_per_stock = portfolio_value / len(predicted_stocks)
    
    current_positions = trader.get_current_positions()
    
    print("\n=== Executing Trades ===")
    
    for ticker in predicted_stocks:
        current_price = trader.get_stock_price(ticker)
        
        if current_price and current_price > 0:
            shares_to_buy = int(allocation_per_stock / current_price)
            
            if ticker in current_positions:
                print(f"Already holding {ticker}: {current_positions[ticker]} shares")
            else:
                if shares_to_buy > 0:
                    trader.place_buy_order(ticker, shares_to_buy)
        else:
            print(f"Could not get price for {ticker}")
    
    print("\n=== Current Positions ===")
    for ticker, qty in current_positions.items():
        print(f"  {ticker}: {qty} shares")


def main():
    """Main function to run strategy and execute trades."""
    
    DRY_RUN = True  # Set to False to execute real trades
    PAPER = True    # Set to False for live trading
    
    print("=" * 50)
    print("MOMENTUM TRADING STRATEGY")
    print("=" * 50)
    mode = "DRY RUN"
    if not DRY_RUN:
        mode = "LIVE " + ("PAPER" if PAPER else "REAL") + " TRADING"
    print(f"Mode: {mode}")
    print("=" * 50)
    
    print("\n[1] Running momentum strategy...")
    top_n, rolling_ret_row, industry_indexed, daily = run_momentum_strategy()
    
    print("\n[2] Predicted stocks for next month:")
    for ticker in top_n.index:
        past_return = rolling_ret_row[ticker]
        sector = industry_indexed.loc[ticker, 'sector'].values[0]
        print(f"  {ticker}: {past_return*100:.2f}% ({sector})")
    
    print("\n[3] Initializing trader...")
    trader = MomentumTrader(dry_run=DRY_RUN, paper=PAPER)
    
    if not DRY_RUN:
        account = trader.get_account_info()
        print(f"Cash: ${account['cash']:.2f}")
        print(f"Buying Power: ${account['buying_power']:.2f}")
    
    print("\n[4] Executing trades...")
    execute_trades(top_n.index.tolist(), trader, portfolio_value=10000)
    
    print("\n[5] Done!")


if __name__ == "__main__":
    main()
