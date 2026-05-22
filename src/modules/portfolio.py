import yfinance as yf
import numpy as np

"""
yahoo finance closing prices.
"""
def closing_prices(assets, start="2020-01-01", end="2026-05-20"):    
    data = yf.download(assets, start=start, end=end)
    
    closing_prices = data["Close"]
    return closing_prices.pct_change().dropna()

def get_covarience(daily_returns):
    covariance_matrix = daily_returns.cov().copy()
    np.fill_diagonal(covariance_matrix.values, 0)

    return covariance_matrix

"""
returns.
"""
def getReturns(allocations, returns):
    return np.dot(returns, allocations)

"""
risk.
"""
def getRisk(covariance, allocations):
    return np.transpose(allocations) @ covariance @ allocations