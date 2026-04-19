import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

def calculate_matrices(G):
    """
    Computes Adjacency, Degree, Laplacian, and Incidence matrices.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Adjacency Matrix (A)
    A = nx.to_numpy_array(G, nodelist=nodes)
    
    # Degree Matrix (D)
    degrees = [val for (node, val) in G.degree(nodes)]
    D = np.diag(degrees)
    
    # Laplacian Matrix (L = D - A)
    L = D - A
    
    # Incidence Matrix (M)
    # NetworkX returns a sparse matrix by default
    M = nx.incidence_matrix(G, nodelist=nodes).toarray()
    
    return A, D, L, M, nodes

def plot_graph_matplotlib(G, title="Graph Visualization"):
    """
    Plots a graph using matplotlib and networkx.
    """
    fig, ax = plt.subplots(figsize=(3, 2.5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='teal', 
            edge_color='gray', node_size=500, font_size=10, 
            font_color='white', ax=ax)
    ax.set_title(title)
    return fig

def get_comp_graph_data(data, node_idx, num_hops=1):
    """
    Extracts a k-hop subgraph for a specific node to show the computational graph.
    """
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, data.edge_index, relabel_nodes=True
    )
    
    # Create a small networkx graph from this subset
    data_sub = Data(edge_index=edge_index, num_nodes=subset.size(0))
    g_sub = to_networkx(data_sub, to_undirected=True)
    return g_sub, subset.tolist()
