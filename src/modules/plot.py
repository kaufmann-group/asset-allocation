import networkx as nx
import numpy as np
import git_root
import matplotlib.pyplot as plt
from matplotlib import colormaps

"""
draws graph of communities
"""
def draw_graph(graph, name=None, labels=None):
    pos = nx.spring_layout(graph, seed=7)
    cmap = colormaps["tab10"]

    node_colors = "lightblue" if labels is None else [cmap(int(c) % 10) for c in labels]

    plt.figure(figsize=(8, 6))

    edges = list(graph.edges(data=True))
    weights = np.array([d.get("weight", 0.0) for _, _, d in edges], dtype=float)
    if len(weights) > 0 and weights.max() != weights.min():
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())
    else:
        norm_weights = np.ones_like(weights) * 0.5

    nx.draw_networkx(graph, pos=pos, node_color=node_colors, with_labels=True, node_size=500, font_size=9, edge_color=norm_weights, edge_cmap=plt.cm.Greys, width=2)

    plt.title("graph")
    plt.axis("off")
    if name is not None:
        plt.savefig(name, dpi=300)
    plt.show()

"""
draws allocations histogram and community diagrams
"""
def plot_allocations_and_communities(graph, assets, weights, communities):

    assets = np.array(assets)
    weights = np.array(weights, dtype=float)
    communities = np.array(communities)

    cmap = colormaps["tab10"]

    unique_communities = sorted(np.unique(communities))

    _, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax_bar, ax_graph = axes

    x_positions = []
    x_labels = []
    bar_heights = []
    bar_colors = []

    x = 0
    gap = 1.0

    for c in unique_communities:
        idx = np.where(communities == c)[0]

        idx = idx[np.argsort(idx)]

        for i in idx:
            x_positions.append(x)
            x_labels.append(f"{assets[i]} ({i})")
            bar_heights.append(weights[i])
            bar_colors.append(cmap(int(c) % 10))
            x += 1

        x += gap

    ax_bar.bar(x_positions, bar_heights, color=bar_colors)

    ax_bar.set_title("Asset Weights and Communities", fontsize=18)
    ax_bar.set_ylabel("Weight", fontsize=14)
    ax_bar.set_xlabel("Assets Grouped by Community", fontsize=14)

    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(x_labels, rotation=90)

    ax_bar.set_ylim(0, max(weights) * 1.15)

    pos = nx.spring_layout(graph)

    node_colors = [cmap(int(communities[node]) % 10) for node in graph.nodes()]

    edges = list(graph.edges(data=True))
    edge_weights = np.array( [d.get("weight", 0.0) for _, _, d in edges], dtype=float )

    if len(edge_weights) > 0 and edge_weights.max() != edge_weights.min():
        norm_edge_weights = ( edge_weights - edge_weights.min() ) / ( edge_weights.max() - edge_weights.min() )
    else:
        norm_edge_weights = np.ones_like(edge_weights) * 0.5

    nx.draw_networkx_nodes(graph,pos,node_color=node_colors,node_size=550,ax=ax_graph)

    nx.draw_networkx_labels(graph,pos,font_size=9,ax=ax_graph)

    nx.draw_networkx_edges(graph,pos,edge_color=norm_edge_weights,edge_cmap=plt.cm.Greys,width=2,alpha=0.65,ax=ax_graph)

    ax_graph.set_title("Asset Communities", fontsize=18)
    ax_graph.axis("off")

    plt.savefig(f"{git_root.git_root()}/data/graph_and_allocations.png", dpi=300)

    plt.tight_layout()
    plt.show()