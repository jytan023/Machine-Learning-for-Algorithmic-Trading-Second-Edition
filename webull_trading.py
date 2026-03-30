"""
Webull Trading Script
=====================
NOTE: The unofficial 'webull' python package only supports PAPER TRADING.
For real trading, consider using: Interactive Brokers, Alpaca, TD Ameritrade, or similar.

This script provides a template for:
- Connecting to Webull
- Getting account info
- Placing buy/sell orders
"""

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

WEBULL_EMAIL = os.getenv("WEBULL_EMAIL")
WEBULL_PASSWORD = os.getenv("WEBULL_PASSWORD")
WEBULL_DEVICE_ID = os.getenv("WEBULL_DEVICE_ID")


def login(webull):
    """Login to Webull with credentials."""
    webull.login(WEBULL_EMAIL, WEBULL_PASSWORD, WEBULL_DEVICE_ID)
    print("Logged in successfully")


def get_account_info(webull):
    """Get account balance and info."""
    account = webull.get_account()
    print(f"Account ID: {account.get('accountId')}")
    print(f"Account Type: {account.get('accountType')}")
    print(f"Cash Balance: ${account.get('cashBalance', 0)}")
    print(f"Buying Power: ${account.get('buyingPower', 0)}")
    return account


def get_positions(webull):
    """Get current positions."""
    positions = webull.get_positions()
    if positions:
        print("\nCurrent Positions:")
        for pos in positions:
            print(f"  {pos.get('symbol')}: {pos.get('quantity')} shares @ ${pos.get('avgCost')}")
    else:
        print("\nNo current positions")
    return positions


def get_stock_quote(webull, ticker):
    """Get current quote for a stock."""
    quote = webull.get_quote(ticker)
    print(f"\n{ticker} Quote:")
    print(f"  Current Price: ${quote.get('close')}")
    print(f"  Open: ${quote.get('open')}")
    print(f"  High: ${quote.get('high')}")
    print(f"  Low: ${quote.get('low')}")
    print(f"  Volume: {quote.get('volume')}")
    return quote


def place_market_order(webull, ticker, quantity, action="BUY"):
    """
    Place a market order.
    
    Args:
        webull: Webull instance
        ticker: Stock symbol (e.g., 'AAPL')
        quantity: Number of shares
        action: 'BUY' or 'SELL'
    
    Returns:
        Order result
    """
    order = webull.place_order(
        stock=ticker,
        action=action,
        orderType="MKT",
        quantity=quantity,
        timeInForce="GTC"  # Good Till Canceled
    )
    print(f"\n{action} Order Placed: {quantity} shares of {ticker}")
    print(f"Order ID: {order.get('orderId')}")
    return order


def place_limit_order(webull, ticker, quantity, price, action="BUY"):
    """
    Place a limit order.
    
    Args:
        webull: Webull instance
        ticker: Stock symbol (e.g., 'AAPL')
        quantity: Number of shares
        price: Limit price
        action: 'BUY' or 'SELL'
    
    Returns:
        Order result
    """
    order = webull.place_order(
        stock=ticker,
        action=action,
        orderType="LMT",
        quantity=quantity,
        limitPrice=price,
        timeInForce="GTC"
    )
    print(f"\n{action} Limit Order Placed: {quantity} shares of {ticker} @ ${price}")
    print(f"Order ID: {order.get('orderId')}")
    return order


def cancel_order(webull, order_id):
    """Cancel an order."""
    result = webull.cancel_order(order_id)
    print(f"\nOrder {order_id} cancelled: {result}")
    return result


def get_orders(webull):
    """Get open orders."""
    orders = webull.get_orders()
    if orders:
        print("\nOpen Orders:")
        for order in orders:
            print(f"  {order.get('symbol')}: {order.get('action')} {order.get('quantity')} @ ${order.get('limitPrice')} ({order.get('orderType')})")
    else:
        print("\nNo open orders")
    return orders


def main():
    """Example usage of Webull trading functions."""
    try:
        from webull import webull
        wb = webull()
        
        # Login
        print("Logging in to Webull...")
        login(wb)
        
        # Get account info
        account = get_account_info(wb)
        
        # Get current positions
        positions = get_positions(wb)
        
        # Example: Get quote for a stock
        # get_stock_quote(wb, 'AAPL')
        
        # Example: Place a market buy order (uncomment to use)
        # place_market_order(wb, 'AAPL', 10, 'BUY')
        
        # Example: Place a limit sell order (uncomment to use)
        # place_limit_order(wb, 'AAPL', 10, 150.00, 'SELL')
        
    except ImportError:
        print("Error: 'webull' package not installed.")
        print("Install with: pip install webull")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
