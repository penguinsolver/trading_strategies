
import sys
import pandas as pd
from datetime import datetime, timedelta
sys.path.insert(0, '.')

from data.candle_fetcher import CandleFetcher

def test_fetch():
    print("Testing data fetch capability...")
    fetcher = CandleFetcher(coin="BTC", use_cache=False)
    
    # Try fetching 11000 candles (approx 450 days)
    # The 'window' argument usually takes '90d' format.
    # Let's try '450d'.
    try:
        print("Attempting to fetch 450d (~11,000 candles)...")
        data = fetcher.fetch_data(interval="1h", window="450d")
        print(f"Fetched: {len(data)} candles")
        print(f"Range: {data.index[0]} to {data.index[-1]}")
        
        if len(data) >= 10000:
            print("SUCCESS: 5 Periods are possible!")
        else:
            print("LIMIT: Could not fetch full history. 5 Periods might be impossible.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fetch()
