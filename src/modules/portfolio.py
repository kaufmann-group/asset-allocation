import yfinance as yf
import numpy as np
import pandas as pd

"""
yahoo finance closing prices
"""
def closing_prices(assets, start="2020-01-01", end="2026-05-20"):    
    data = yf.download(assets, start=start, end=end) # needs numpy array? 
    
    closing_prices = data["Close"]
    return closing_prices.pct_change().dropna()

"""
gets covariance matrix from daily returns
"""
def get_covariance(daily_returns):
    # 1. Calculate the covariance matrix
    covariance_matrix = daily_returns.cov()
    
    # 2. Create a matching boolean DataFrame mask for the diagonal
    diagonal_mask = pd.DataFrame(
        np.eye(covariance_matrix.shape[0], dtype=bool),
        index=covariance_matrix.index,
        columns=covariance_matrix.columns
    )
    
    # 3. Mask the diagonal elements with 0
    covariance_matrix = covariance_matrix.mask(diagonal_mask, 0)
    
    return covariance_matrix
"""
gets correlation matrix from daily returns
"""
def get_correlation(daily_returns):
    return daily_returns.corr().abs()

"""
returns
"""
def getReturns(allocations, returns):
    return np.dot(returns, allocations)

"""
risk
"""
def getRisk(covariance, allocations):
    return np.transpose(allocations) @ covariance @ allocations