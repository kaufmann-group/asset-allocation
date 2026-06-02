import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from modules import *
from main import community_asset_allocation

if __name__ == "__main__":

    number_runs=200
    number_communities=5

    assets_dict = parse_assets_file("benchmarking_assets.txt")
    choose = lambda area, n : np.random.choice(assets_dict[area], n, replace=False).tolist()

    assets = choose("Technology & Semiconductors", 5) + choose("Communication Services & Media", 5) + choose("Consumer Discretionary & Retail", 5) + choose("Industrials, Energy & Utilities", 5)

    """
    benchmarking
    """

    daily_returns = closing_prices(assets=assets)

    returns = daily_returns.mean() * 252 # returns 
    cov_matrix = get_covariance(daily_returns=daily_returns, annualize=True) # covariance matrix

    cov_rar = []
    corr_rar = []

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities, adjacency=get_covariance)
        corr = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities, adjacency=get_correlation)

        cov_rar.append((get_risk(covariance=cov_matrix, allocations=caa), get_returns(allocations=caa, returns=returns)))
        corr_rar.append((get_risk(covariance=cov_matrix, allocations=corr), get_returns(allocations=corr, returns=returns)))

    """
    plotting
    """

    figure_2, ax = plt.subplots(figsize=(7, 5))

    ax.plot(*zip(*cov_rar), "bo", label="Covariance Graph")
    ax.plot(*zip(*corr_rar), "ro", label="Correlation Graph")

    ax.set_xlabel("Risk", fontsize=11) 
    ax.set_ylabel("Return", fontsize=11)
    ax.set_title("Covariance vs Correlations as Adjacency for Community Detection", fontsize=12)

    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("../../figures/adjacency_benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()