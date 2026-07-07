"""
Webull Trading Script (Official SDK)
====================================
Uses the official Webull OpenAPI SDK for live trading.

Installation:
    pip install webull-python-sdk-core webull-python-sdk-trade
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

WEBULL_EMAIL = os.getenv("WEBULL_EMAIL")
WEBULL_PASSWORD = os.getenv("WEBULL_PASSWORD")
WEBULL_DEVICE_ID = os.getenv("WEBULL_DEVICE_ID")
WEBULL_ACCOUNT_ID = os.getenv("WEBULL_ACCOUNT_ID")


class WebullTrader:
    def __init__(self, paper=True):
        """
        Initialize Webull trader.
        
        Args:
            paper: If True, use paper trading account. If False, use real trading.
        """
        from webull import webull
        from webull.trade import webull_trade
        
        self.wb = webull()
        self.paper = paper
        
        if paper:
            self.trade = webull_trade(self.wb, paper_account=True)
            print("Connected to Webull Paper Trading")
        else:
            self.trade = webull_trade(self.wb, paper_account=False)
            print("Connected to Webull Live Trading")
    
    def login(self):
        """Login to Webull."""
        self.wb.login(WEBULL_EMAIL, WEBULL_PASSWORD, WEBULL_DEVICE_ID)
        self.trade.get_account_id()  # Initialize account
        print("Logged in successfully")
    
    def get_account(self):
        """Get account info."""
        return self.trade.get_account()
    
    def get_positions(self):
        """Get current positions."""
        return self.trade.get_positions()
    
    def get_quote(self, ticker):
        """Get stock quote."""
        return self.wb.get_quote(ticker)
    
    def place_market_order(self, ticker, quantity, action="BUY"):
        """
        Place a market order.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares
            action: 'BUY' or 'SELL'
        
        Returns:
            Order result
        """
        return self.trade.place_order(
            stock=ticker,
            action=action,
            orderType="MKT",
            quantity=quantity,
            timeInForce="GTC"
        )
    
    def place_limit_order(self, ticker, quantity, price, action="BUY"):
        """Place a limit order."""
        return self.trade.place_order(
            stock=ticker,
            action=action,
            orderType="LMT",
            quantity=quantity,
            limitPrice=price,
            timeInForce="GTC"
        )
    
    def cancel_order(self, order_id):
        """Cancel an order."""
        return self.trade.cancel_order(order_id)
    
    def get_open_orders(self):
        """Get open orders."""
        return self.trade.get_open_orders()


def main():
    """Example usage."""
    try:
        trader = WebullTrader(paper=True)  # Set paper=False for live trading
        
        print("Logging in...")
        trader.login()
        
        account = trader.get_account()
        print(f"\nAccount Balance: ${account.get('cashBalance', 0)}")
        print(f"Buying Power: ${account.get('buyingPower', 0)}")
        
        positions = trader.get_positions()
        print(f"\nPositions: {len(positions)}")
        
        # Example: Get quote
        # quote = trader.get_quote('AAPL')
        # print(f"\nAAPL Price: ${quote.get('close')}")
        
        # Example: Place order
        # order = trader.place_market_order('AAPL', 10, 'BUY')
        # print(f"\nOrder placed: {order}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
