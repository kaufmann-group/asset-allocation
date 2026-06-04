import yfinance as yf
import numpy as np
import pandas as pd

"""
yahoo finance closing prices
"""
def closing_prices(assets, start="2025-04-04", end="2026-04-04"):    
    data = yf.download(assets, start=start, end=end) # needs numpy array? 
    
    closing_prices = data["Close"]
    return closing_prices.pct_change().dropna()

"""
gets covariance matrix from daily returns
"""
def get_covariance(daily_returns, annualize=False, zero_diagonal=False):
    covariance_matrix = daily_returns.cov() * (252 if annualize else 1)
    
    if zero_diagonal:
        diagonal_mask = pd.DataFrame(np.eye(covariance_matrix.shape[0], dtype=bool), index=covariance_matrix.index, columns=covariance_matrix.columns)
        covariance_matrix = covariance_matrix.mask(diagonal_mask, 0)
    
    return covariance_matrix
"""
gets correlation matrix from daily returns
"""
def get_correlation(daily_returns, positive_correlation_communities=False, zero_diagonal=False):
    corr = daily_returns.corr().clip(lower=0) if positive_correlation_communities else daily_returns.corr().abs()

    if zero_diagonal:
        corr_matrix = corr.values.copy()
        np.fill_diagonal(corr_matrix, 0.0)
        corr = corr.__class__(corr_matrix, index=corr.index, columns=corr.columns)

    return corr

"""
returns
"""
def get_returns(allocations, returns):
    return np.dot(returns, allocations)

"""
risk
"""
def get_risk(covariance, allocations):
    return np.transpose(allocations) @ covariance @ allocations

"""
sharpe ratio
"""
def get_sharpe_ratio(allocations, returns, covariance, risk_free_rate=0.0):
    port_return = get_returns(allocations, returns)
    
    port_variance = get_risk(covariance, allocations)
    port_standard_deviation = np.sqrt(port_variance)

    if port_standard_deviation == 0 or np.isnan(port_standard_deviation):
        return np.nan
    else:
        return (port_return - risk_free_rate) / port_standard_deviation

"""
calculates the Herfindahl-Hirschman Index for diversification
"""
def get_hhi(weights):
    return np.linalg.norm(weights) ** 2

"""
calculates diversification ratio
"""

def get_diversification_ratio(weights, covariance):
    asset_vols = np.sqrt(np.diag(covariance))
    weighted_avg_vol = weights @ asset_vols

    portfolio_var = weights @ covariance @ weights
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol == 0 or np.isnan(portfolio_vol):
        return np.nan
    else:
        return weighted_avg_vol / portfolio_vol

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