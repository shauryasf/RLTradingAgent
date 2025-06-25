"""
Simple fallback environment when FinRL is not available
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class SimpleTradingEnv:
    """
    Minimal trading environment that works without FinRL dependencies
    """
    
    def __init__(self, ticker=None, start_date=None, end_date=None, **kwargs):
        self.ticker = ticker
        self.df = None
        
        if ticker:
            self.df = self.get_simple_data(ticker, start_date, end_date)
        elif 'df' in kwargs:
            self.df = kwargs['df']
        else:
            # Create dummy data
            self.df = self.create_dummy_data(ticker or "DEMO")
    
    def get_simple_data(self, ticker, start_date, end_date):
        """Get stock data using yfinance with fallback to dummy data"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Convert to simple format
            df = data.reset_index()
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
            df['tic'] = ticker
            return df
            
        except Exception as e:
            print(f"Failed to download {ticker}: {e}. Using dummy data...")
            return self.create_dummy_data(ticker)
    
    def create_dummy_data(self, ticker):
        """Create realistic dummy data"""
        if not hasattr(self, '_dummy_data_cache'):
            # Create 1000 days of dummy data
            dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
            
            # Generate realistic price movements
            np.random.seed(42)  # For reproducibility
            base_price = 100.0
            returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
            
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1.0))  # Ensure positive prices
            
            # Create OHLCV data
            opens = prices[:-1] + [prices[-1]]
            closes = prices
            highs = [max(o, c) * (1 + np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
            lows = [min(o, c) * (1 - np.random.uniform(0, 0.01)) for o, c in zip(opens, closes)]
            volumes = [1000000 * (1 + np.random.uniform(-0.3, 0.3)) for _ in range(len(dates))]
            
            self._dummy_data_cache = pd.DataFrame({
                'date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes,
                'tic': ticker
            })
        
        return self._dummy_data_cache.copy()
    
    def reset(self):
        """Reset environment (minimal gym interface)"""
        if self.df is not None and len(self.df) > 0:
            # Return basic observation (just the latest close price for now)
            obs = np.array([self.df['close'].iloc[-1]])
            return obs, {}
        else:
            return np.array([100.0]), {}  # Default observation
