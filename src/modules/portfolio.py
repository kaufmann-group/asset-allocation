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
    covariance_matrix = daily_returns.cov()
    
    diagonal_mask = pd.DataFrame( np.eye(covariance_matrix.shape[0], dtype=bool), index=covariance_matrix.index, columns=covariance_matrix.columns )
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

"""
sharpe ratio
"""
def getSharpeRatio(allocations, returns, covariance, risk_free_rate=0.0):
    port_return = getReturns(allocations, returns)
    
    port_variance = getRisk(covariance, allocations)
    port_standard_deviation = np.sqrt(port_variance)
    
    sharpe_ratio = (port_return - risk_free_rate) / port_standard_deviation
    
    return sharpe_ratio

"""
parses assets text file
"""
def parse_assets_file(filepath):
    asset_dict = {}
    current_category = None

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith("------------"):
                current_category = line.replace("-", "").strip()
                asset_dict[current_category] = []
                continue
            
            if current_category:
                ticker = line.split()[0]
                asset_dict[current_category].append(ticker)

    return asset_dict