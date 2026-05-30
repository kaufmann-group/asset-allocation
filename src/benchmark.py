"""
benchmarking community biased asset allocation.
"""

from tqdm import tqdm

from modules import *
from main import community_asset_allocation

"""
this test runs asset allocation on 20 assets against community
detection based asset allocation on 20 assets.
"""
def benchmark_1(assets, number_runs=200, number_communities=5):
    daily_returns = closing_prices(assets=assets)

    returns = daily_returns.mean() * 252 # returns
    cov_matrix = get_covariance(daily_returns=daily_returns) # covariance matrix

    caa_rar = [] # community asset allocation risk and returns
    aa_rar = [] # asset allocation risk and returns

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities) # community asset allocations

        caa_rar.append((getRisk(covariance=cov_matrix, allocations=caa), getReturns(allocations=caa, returns=returns)))

        aa = AssetAllocation(returns=returns.to_numpy(), covariance=cov_matrix.to_numpy()) # asset allocation object 
        aas = aa.run(solver_type = "SIMULATED") # asset allocations
     

        aa_rar.append((getRisk(covariance=cov_matrix, allocations=aas), getReturns(allocations=aas, returns=returns)))

    return aa_rar, caa_rar

"""
this test runs community asset allocation 
"""
def benchmark_2(assets, number_runs=200, number_communities=5):
    daily_returns = closing_prices(assets=assets)

    returns = daily_returns.mean() * 252 # returns 
    cov_matrix = get_covariance(daily_returns=daily_returns) # covariance matrix

    cov_rar = []
    corr_rar = []

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities)
        corr = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities)

        cov_rar.append((getRisk(covariance=cov_matrix, allocations=caa), getReturns(allocations=caa, returns=returns)))
        corr_rar.append((getRisk(covariance=cov_matrix, allocations=corr), getReturns(allocations=corr, returns=returns)))

    return cov_rar, corr_rar

"""
test diversification
"""



if __name__ == "__main__":
    with open("benchmarking_assets.txt", "r") as file:
        total_assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        random_indices = np.random.choice(len(total_assets), size=20, replace=False)
        assets = [total_assets[i] for i in random_indices]

        """
        first benchmarking test
        """

        asset_allocations, community_asset_allocations = benchmark_1(assets=assets, number_communities=3) # three communities

        plt.plot(*zip(*asset_allocations), "bo", label="without communities")
        plt.plot(*zip(*community_asset_allocations), "ro", label="with communities")

        plt.xlabel("Risk") 
        plt.ylabel("Return")
        plt.title("Regular Asset Allocations vs Community Based Asset Allocation")
        plt.legend()

        plt.savefig("../figures/benchmark_1.png", dpi=300)
        plt.show()

        """
        second benchmarking test
        """

        community_asset_allocation_with_correlations, community_asset_allocation_with_covariance = benchmark_2(assets=assets, number_communities=3)

        plt.plot(*zip(*community_asset_allocation_with_correlations), "bo", label="correlations graph")
        plt.plot(*zip(*community_asset_allocation_with_covariance), "ro", label="covariance graph")

        plt.xlabel("Risk") 
        plt.ylabel("Return")
        plt.title("Covariance vs Correlations as Adjacency for Community Detection")
        plt.legend()

        plt.savefig("../figures/benchmark_2.png", dpi=300)
        plt.show()




