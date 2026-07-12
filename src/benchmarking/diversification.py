import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
import git_root
sys.path.append("..")

from modules import *
from main import community_asset_allocation

"""
test whether the our community based asset allocation algorithm produces a more genuinely 
diversified portfolio than the classical allocation method as the number of available assets increases.
"""

if __name__ == "__main__":

    number_runs = 10
    number_communities = 5

    with open("benchmarking_assets.txt", "r") as file:
        all_assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

    """
    benchmarking
    """

    num_assets = np.arange(20, 55, 5)

    aa_metrics = np.zeros((len(num_assets), 2))
    caa_metrics = np.zeros((len(num_assets), 2))

    for idx, n in enumerate(tqdm(num_assets, desc="Benchmarking Assets")):
            assets = all_assets[:n]

            daily_returns = closing_prices(assets=assets)
            returns = daily_returns.mean() * 252
            cov_matrix = get_covariance(daily_returns=daily_returns, annualize=True)

            run_aa_div, run_aa_hhi = [], []
            run_caa_div, run_caa_hhi = [], []

            for _ in range(number_runs):
                caa_weights = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities)
                run_caa_div.append(get_diversification_ratio(weights=caa_weights, covariance=cov_matrix))
                run_caa_hhi.append(get_hhi(weights=caa_weights))

                aa = AssetAllocation(returns=returns.to_numpy(), covariance=cov_matrix.to_numpy())
                aa_weights = aa.run(solver_type="SIMULATED")
                run_aa_div.append(get_diversification_ratio(weights=aa_weights, covariance=cov_matrix))
                run_aa_hhi.append(get_hhi(weights=aa_weights))

            aa_metrics[idx] = [np.mean(run_aa_div), np.mean(run_aa_hhi)]
            caa_metrics[idx] = [np.mean(run_caa_div), np.mean(run_caa_hhi)]

    """
    plotting
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    fig.suptitle("Diversification Metrics", fontsize=14, y=1.00)

    axes[0].plot( num_assets, aa_metrics[:, 0], label="Classical AA", marker="o", linewidth=2)
    axes[0].plot( num_assets, caa_metrics[:, 0], label="Community CAA", marker="s", linewidth=2)
    axes[0].set_title("Diversification Ratio Comparison", fontsize=12, pad=10)
    axes[0].set_ylabel("Diversification Ratio")
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot( num_assets, aa_metrics[:, 1], label="Classical AA", marker="o", linewidth=2)
    axes[1].plot(num_assets, caa_metrics[:, 1], label="Community CAA", marker="s", linewidth=2)
    axes[1].set_title("Portfolio Concentration (HHI)", fontsize=12, pad=10)
    axes[1].set_ylabel("HHI Score")
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    for ax in axes:
        ax.set_xlabel("Number of Assets")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{git_root.git_root()}/data/diversification_benchmark.png", dpi=300)
    plt.show()

    """
    save benchmarking data to csv
    """

    data = {
        "Number of Assets": num_assets,
        "Classical Asset Allocation Diversification Ratio": aa_metrics[:, 0],
        "Classical Asset Allocation HHI": aa_metrics[:, 1],
        "Community Asset Allocation Diversification Ratio": caa_metrics[:, 0],
        "Community Asset Allocation HHI": caa_metrics[:, 1]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(f"{git_root.git_root()}/data/diversification_benchmark.csv", index=False)