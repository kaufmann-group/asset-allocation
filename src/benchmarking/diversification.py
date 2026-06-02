import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from modules import *
from main import community_asset_allocation

if __name__ == "__main__":

    number_runs=3
    number_communities=5

    with open("benchmarking_assets.txt", "r") as file:
        all_assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

    """
    benchmarking
    """

    num_assets = np.arange(20, 95, 5)

    aa_diversification = []
    caa_diversification = []

    for n in tqdm(num_assets):
        assets = all_assets[:n]

        daily_returns = closing_prices(assets=assets)
        returns = daily_returns.mean() * 252 # returns

        cov_matrix = get_covariance(daily_returns=daily_returns, annualize=True)
    
        aa_diversifications = []
        caa_diversifications = []

        for _ in range(number_runs):
            caa_weights = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities)
            caa_diversifications.append([get_diversification_ratio(weights=caa_weights, covariance=cov_matrix), get_hhi(weights=caa_weights)])

            aa = AssetAllocation(returns=returns.to_numpy(), covariance=cov_matrix.to_numpy())
            aa_weights = aa.run(solver_type="SIMULATED")
            aa_diversifications.append([get_diversification_ratio(weights=aa_weights, covariance=cov_matrix), get_hhi(weights=aa_weights)])

        aa_diversification.append(np.mean(aa_diversifications, axis=0))
        caa_diversification.append(np.mean(caa_diversifications, axis=0))

    """
    plotting
    """

    figure, axis = plt.subplots(1, 2, figsize=(12, 5))

    axis[0].plot(num_assets, aa_diversification[:, 0], label="AA diversification ratio")
    axis[0].plot(num_assets, caa_diversification[:, 0], label="CAA diversification ratio")

    axis[1].plot(num_assets, aa_diversification[:, 1], label="AA HHI")
    axis[1].plot(num_assets, caa_diversification[:, 1], label="CAA HHI")

    plt.show()



