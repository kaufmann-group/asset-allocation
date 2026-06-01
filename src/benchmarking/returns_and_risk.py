import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import plt

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

    caa_rar = [] # community asset allocation risk and returns
    aa_rar = [] # asset allocation risk and returns
    sharpe_ratios = [] # sharpe ratios

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities) # community asset allocations

        caa_rar.append((get_risk(covariance=cov_matrix, allocations=caa), get_returns(allocations=caa, returns=returns)))

        aa = AssetAllocation(returns=returns.to_numpy(), covariance=cov_matrix.to_numpy()) # asset allocation object 
        aas = aa.run(solver_type = "SIMULATED") # asset allocations
     

        aa_rar.append((get_risk(covariance=cov_matrix, allocations=aas), get_returns(allocations=aas, returns=returns)))

        sharpe_ratios.append((get_sharpe_ratio(allocations=caa, returns=returns, covariance=cov_matrix), get_sharpe_ratio(allocations=aas, returns=returns, covariance=cov_matrix)))


    """
    plotting
    """

    figure_1, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure_1.suptitle("Asset Allocation vs Community Based Asset Allocation", fontsize=14, y=1.00)

    axes[0].plot(*zip(*aa_rar), "bs", markersize=5, label="Without Communities")
    axes[0].plot(*zip(*caa_rar), "r^", markersize=6, label="With Communities")

    axes[0].set_xlabel("Risk", fontsize=11)
    axes[0].set_ylabel("Return", fontsize=11)
    axes[0].set_title("Risk vs Returns", fontsize=12)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(*zip(*sharpe_ratios), "r*", label="Sharpe Comparison")
    axes[1].axline((0, 0), (1, 1), color="k", linestyle=":", linewidth=1, transform=axes[1].transAxes)

    axes[1].set_xlabel("Sharpe Ratio: Community Algorithm", fontsize=11)
    axes[1].set_ylabel("Sharpe Ratio: Asset Allocation", fontsize=11)
    axes[1].set_title("Sharpe Ratios for Both Algorithms", fontsize=12)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("../../figures/benchmark_1.png", dpi=300, bbox_inches="tight")
    plt.show()
