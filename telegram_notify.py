"""
Telegram Notification Module
=============================
Sends messages to Telegram using the Bot API.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_message(text):
    """
    Send a message to Telegram.
    
    Args:
        text: Message text to send
    
    Returns:
        True if successful, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials not set. Skipping notification.")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Telegram message sent successfully")
            return True
        else:
            print(f"Failed to send Telegram message: {response.text}")
            return False
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False


def format_stock_message(top_n, rolling_ret_row, industry_indexed, month_str, daily=None, l=1.07):
    """
    Format the stock prediction message for Telegram.
    
    Args:
        top_n: DataFrame with top stocks
        rolling_ret_row: Series with past returns
        industry_indexed: DataFrame with sector info
        month_str: String for the month
        daily: DataFrame with daily price data (optional)
        l: Limit/take profit threshold
    
    Returns:
        Formatted message string
    """
    message = f"<b>📈 Momentum Stocks - {month_str}</b>\n\n"
    message += "<b>Top 3 Picks (Past 3-Month Return):</b>\n"
    
    for i, ticker in enumerate(top_n.index, 1):
        past_return = rolling_ret_row[ticker]
        sector = industry_indexed.loc[ticker, 'sector']
        sector_val = sector.values[0] if hasattr(sector, 'values') else sector
        emoji = "🟢" if past_return > 0 else "🔴"
        
        message += f"{i}. {emoji} <b>{ticker}</b> - {past_return*100:.2f}%\n"
        message += f"   📊 {sector_val}\n"
        
        if daily is not None:
            buy_price = daily['Adj Close'][ticker].iloc[-1]
            sell_price = buy_price * l
            message += f"   💰 Buy @ ${buy_price:.2f} | 🎯 Sell @ ${sell_price:.2f}\n"
        
        message += "\n"
    
    message += f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    return message


def send_stock_prediction(top_n, rolling_ret_row, industry_indexed, month_str, daily=None, l=1.07):
    """Send stock prediction to Telegram."""
    message = format_stock_message(top_n, rolling_ret_row, industry_indexed, month_str, daily, l)
    return send_message(message)


if __name__ == "__main__":
    # Test message
    send_message("🤖 Test: Telegram notification is working!")