import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_networkx


def visualize_rewired_graph(i, graphs_pyg_list, rewired_graph_list, ensemble_size):
    """
    Visualize the rewired graph for molecule 'i' by comparing its original, added, and deleted edges.

    Args:
        i: Index of the molecule in the batch.
        graphs_pyg_list: List of original graphs (PyTorch Geometric Data objects).
        rewired_graph_list: List of rewired graphs (PyTorch Geometric Data objects).
        ensemble_size: Number of ensemble runs.
    """
    # Initialize sets and dictionaries to store edges
    original_edges_set = set()
    added_edges = {}
    deleted_edges_count = {}

    # Determine batch size based on ensemble_size
    batch_size = len(graphs_pyg_list)

    # Extract the ensemble graphs for molecule 'i'
    ensemble_graphs = []
    for j in range(ensemble_size):
        idx = i + j * batch_size
        if idx < len(rewired_graph_list):
            ensemble_graphs.append(rewired_graph_list[idx])
        else:
            print(f"Warning: Index {idx} out of bounds for rewired_graph_list with length {len(rewired_graph_list)}")
            continue

    # Get the original graph for molecule 'i'
    original_graph = graphs_pyg_list[i]
    original_edge_index = original_graph.edge_index.t().tolist()  # Transpose and convert to list of edges

    # Collect original edges (as sorted tuples for undirected graphs)
    for edge in original_edge_index:
        edge = tuple(sorted(edge))
        original_edges_set.add(edge)

    # Initialize a set to collect all edges from the rewired graphs
    all_rewired_edges = set()

    # Iterate over the ensemble graphs to accumulate edge additions and deletions
    for g in ensemble_graphs:
        edge_index = g.edge_index.t().tolist()  # List of edges
        edge_weight = g.edge_weight.cpu().numpy()  # Edge weights

        rewired_edges_set = set()
        for idx, (src, dest) in enumerate(edge_index):
            edge = tuple(sorted([src, dest]))
            rewired_edges_set.add(edge)

            if edge in original_edges_set:
                # If the edge is in the original graph, check if it's deleted
                # Assuming that edge_weight != 1.0 signifies deletion or modification
                # Here, we'll treat any deviation from 1.0 as a deletion
                if edge_weight[idx] < 1.0:  # Threshold can be adjusted as needed
                    deleted_edges_count[edge] = deleted_edges_count.get(edge, 0) + 1
            else:
                # Edge is added
                added_edges[edge] = added_edges.get(edge, 0) + edge_weight[idx]

        # Collect all edges from this rewired graph
        all_rewired_edges.update(rewired_edges_set)

    # Detect deleted edges: edges present in original but not in any rewired graph
    for edge in original_edges_set:
        if edge not in all_rewired_edges:
            # Edge is completely deleted in all ensemble runs
            deleted_edges_count[edge] = deleted_edges_count.get(edge, 0) + ensemble_size

    # Create the visualization graph using NetworkX
    G = nx.Graph()

    # Add original edges that were not deleted (black)
    for edge in original_edges_set:
        if edge not in deleted_edges_count:
            G.add_edge(edge[0], edge[1], color='black', weight=1.0)

    # Add deleted edges (red) with thickness based on deletion frequency
    for edge, count in deleted_edges_count.items():
        # Calculate deletion frequency as a proportion of ensemble runs
        deletion_frequency = count / ensemble_size
        G.add_edge(edge[0], edge[1], color='red', weight=deletion_frequency)

    # Add added edges (green) with thickness based on average weight
    for edge, total_weight in added_edges.items():
        avg_weight = total_weight / ensemble_size  # Average weight over ensemble
        G.add_edge(edge[0], edge[1], color='green', weight=avg_weight)

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for reproducible layout
    edges = G.edges(data=True)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges with different colors and thicknesses
    for edge in edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(edge[0], edge[1])],
            edge_color=edge[2]['color'],
            width=edge[2]['weight'] * 2  # Adjust scaling factor as needed
        )

    # Draw labels
    # nx.draw_networkx_labels(G, pos, font_size=10)

    # plt.title(f"Molecule {i} Rewiring Visualization")
    plt.axis('off')  # Hide axis

    file_path = f"figs/molecule_{i}_rewiring_visualization.pdf"  # Use .svg or .pdf
    plt.savefig(file_path, format='pdf')  # You can also use 'pdf' if preferred

    plt.show()
    plt.clf()  # Clear the current figure to avoid overlap if generating multiple graphs

