"""
program runs community biased asset allocation on predefined stocks.
"""

import numpy as np
import networkx as nx

from modules import *

lambda_3 = 50.0   # penalize risk more
lambda_1 = 0.2    # reward return less aggressively

def community_asset_allocation(daily_returns, number_communities, solver_type="SIMULATED", adjacency=get_covariance):
    returns = daily_returns.mean() * 252 

    covariance_matrix = get_covariance(daily_returns=daily_returns, annualize=True)

    graph_adjacency = adjacency(daily_returns=daily_returns, zero_diagonal=True)

    community_detection = CommunityDetection(adjacency_matrix=graph_adjacency, number_communities=number_communities)
    communities = community_detection.run(solver_type=solver_type)
    partitions = [set(np.where(communities == c)[0]) for c in np.unique(communities)] 

    group_average_returns = {} 
    group_daily_returns = np.zeros((len(daily_returns), len(partitions))) 

    """
    note this converts each community/partition into a single "asset" by computing its average annual return and
    its daily return time series then partition_covariance_matrix computes the covariance
    """
    for group_index, asset_group in enumerate(partitions):
        asset_group = list(asset_group)

        group_average_returns[group_index] = returns.iloc[asset_group].mean()
        group_daily_returns[:, group_index] = daily_returns.iloc[:, asset_group].mean(axis=1).to_numpy()

    partition_covariance_matrix = np.cov(group_daily_returns, rowvar=False) * 252 

    upper_returns = np.array(list(group_average_returns.values())) # note that upper community returns are computed as equal-weight community returns what to do abt this ????
    upper_allocation = AssetAllocation(returns=upper_returns, covariance=partition_covariance_matrix, lambda_1=lambda_1, lambda_3=lambda_3)
    upper_allocations = upper_allocation.run(solver_type=solver_type)

    lower_allocations = []

    for _, cluster in enumerate(partitions):
        cluster = list(cluster)

        cluster_returns = [returns.iloc[asset] for asset in cluster]
        cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

        inner_allocation = AssetAllocation(returns=cluster_returns, covariance=cluster_covariance, lambda_1=lambda_1, lambda_3=lambda_3) 
        lower_allocations.append(inner_allocation.run(solver_type=solver_type))

    allocations = np.zeros(len(returns))
    for group_weight, cluster, inner_weights in zip(upper_allocations, partitions, lower_allocations):
        cluster = list(cluster)

        for asset_idx, inner_weight in zip(cluster, inner_weights):
            allocations[asset_idx] = group_weight * inner_weight

    # plot_allocations_and_communities(graph=graph, assets=assets, weights=allocations, communities=communities)

    return allocations

if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        daily_returns = closing_prices(assets=assets)
        allocations = community_asset_allocation(daily_returns=daily_returns, number_communities=4)

        for asset, allocation in zip(assets, allocations):
            print(f"{asset}: {allocation}")

        