import torch
from torch.xpu import device
from torch_geometric.data import Data

def transform_pyg_batch(repeated_data, pyg_batch, num_features=20, is_QM9=False, is_ZINC=False):

    assert len(pyg_batch) == 1, "We are only handling the case with one batch of graphs."
    pyg_batch = pyg_batch[0]
    graph_list = manually_split_graphs_general(pyg_batch)
    # graph_list = pyg_batch.to_data_list()
    # graph_list = extract_and_assign_edge_weights(pyg_batch, graph_list)

    transformed_graphs = [
        transform_graph(instance, pyg_graph, num_features=num_features, is_QM9=is_QM9, is_ZINC=is_ZINC)
        for instance, pyg_graph in zip(repeated_data, graph_list)
    ]

    # transformed_graphs = [transform_graph(instance, pyg_graph, num_features=num_features, is_QM9=is_QM9, is_ZINC=is_ZINC)
    #                       for instance, pyg_graph in zip(repeated_data, graph_list)]

    batched_features = torch.stack(transformed_graphs, dim=0)

    return batched_features


def transform_graph(instance, pyg_graph, num_features=20, is_QM9=False, is_ZINC=False):

    assert pyg_graph.num_nodes == pyg_graph.x.size(0), "Number of nodes and node features do not match."

    num_nodes = pyg_graph.num_nodes
    graph_features = torch.zeros(num_features, num_nodes, num_nodes, device=pyg_graph.x.device, dtype=pyg_graph.x.dtype)

    # Efficiently create diagonal matrices for node features
    diags = torch.diag_embed(pyg_graph.x.T)  # Embedding the first 13 (or 25) features

    if is_QM9:
        graph_features[:13] = diags

        # Additional features
        if hasattr(pyg_graph, 'distance_mat'):
            graph_features[13] = pyg_graph.distance_mat
        if hasattr(pyg_graph, 'affinity'):
            graph_features[14] = pyg_graph.affinity
        if hasattr(pyg_graph, 'edge_features'):
            graph_features[15:19] = pyg_graph.edge_features.view(4, num_nodes, num_nodes)  # Adjust based on structure
        if hasattr(pyg_graph, 'edge_weight') and pyg_graph.edge_weight is not None:
            edge_weight = pyg_graph.edge_weight
            edge_index = pyg_graph.edge_index
            num_nodes = pyg_graph.num_nodes  # Ensure you have the correct number of nodes

            # Move tensors to the same device
            device = edge_weight.device
            adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)

            # Vectorized assignment
            adjacency_matrix[edge_index[0], edge_index[1]] = edge_weight
            # adjacency_matrix[edge_index[1], edge_index[0]] = edge_weight  # For undirected graphs

            graph_features[19] = adjacency_matrix


    elif is_ZINC:
        graph_features[:21] = diags

        if hasattr(pyg_graph, 'edge_features'):
            graph_features[21:25] = pyg_graph.edge_features.view(4, num_nodes, num_nodes)
        else:
            graph_features[21:25] = instance[-4:, :, :]  # Use the last 4 features from the instance  TODO: Get these from the pyg_graph

        if hasattr(pyg_graph, 'edge_weight') and pyg_graph.edge_weight is not None:
            edge_weight = pyg_graph.edge_weight
            edge_index = pyg_graph.edge_index
            num_nodes = pyg_graph.num_nodes  # Ensure you have the correct number of nodes

            # Move tensors to the same device
            device = edge_weight.device
            adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)

            # Vectorized assignment
            adjacency_matrix[edge_index[0], edge_index[1]] = edge_weight
            # adjacency_matrix[edge_index[1], edge_index[0]] = edge_weight  # For undirected graphs

            graph_features[25] = adjacency_matrix


    return graph_features


# def manually_split_graphs_general(pyg_batch):
#     """
#     Manually splits a batched graph into individual graphs, handling additional properties dynamically.
#
#     Args:
#         pyg_batch: A PyTorch Geometric DataBatch object containing multiple graphs with various properties.
#
#     Returns:
#         A list of individual graph Data objects, including all additional attributes.
#     """
#     graph_list = []
#
#     for graph_index in range(pyg_batch.num_graphs):
#         node_start = pyg_batch.ptr[graph_index].item()
#         node_end = pyg_batch.ptr[graph_index + 1].item()
#         num_nodes = node_end - node_start
#         node_indices = torch.arange(node_start, node_end, device=pyg_batch.x.device)
#
#         # Create a mapping from global node indices to local node indices
#         global_to_local = torch.full((pyg_batch.x.size(0),), -1, dtype=torch.long, device=pyg_batch.x.device)
#         global_to_local[node_indices] = torch.arange(num_nodes, device=pyg_batch.x.device)
#
#         # Find edges that belong to the current graph
#         edge_mask = (pyg_batch.edge_index[0] >= node_start) & (pyg_batch.edge_index[0] < node_end) & \
#                     (pyg_batch.edge_index[1] >= node_start) & (pyg_batch.edge_index[1] < node_end)
#
#         # Extract and adjust edge_index
#         edge_index = pyg_batch.edge_index[:, edge_mask]
#         edge_index = global_to_local[edge_index]
#
#         # Create a new Data object for the current graph
#         graph = Data()
#
#         # Dynamically assign the attributes in the batch to the graph
#         for key, value in pyg_batch:
#             if key in ['edge_index', 'batch', 'ptr']:
#                 continue  # Skip attributes that are not graph-specific
#
#             # Handle edge-related attributes
#             if torch.is_tensor(value) and value.size(0) == pyg_batch.edge_index.size(1):
#                 graph[key] = value[edge_mask]
#
#             # Handle node-related attributes
#             elif torch.is_tensor(value) and value.size(0) == pyg_batch.x.size(0):
#                 graph[key] = value[node_indices]
#
#             # Handle graph-level attributes (single value per graph)
#             elif torch.is_tensor(value) and value.size(0) == pyg_batch.num_graphs:
#                 graph[key] = value[graph_index]
#
#             else:
#                 # Handle other attributes if necessary
#                 pass
#
#         # Set the adjusted edge_index for the graph
#         graph.edge_index = edge_index
#
#         # Remove the batch attribute from the individual graph
#         if hasattr(graph, 'batch'):
#             del graph.batch
#
#         graph_list.append(graph)
#
#     return graph_list


def manually_split_graphs_general(pyg_batch):
    """
    Manually splits a batched graph into individual graphs, handling additional properties dynamically.

    Args:
        pyg_batch: A PyTorch Geometric DataBatch object containing multiple graphs with various properties.

    Returns:
        A list of individual graph Data objects, including all additional attributes.
    """
    graph_list = []

    for graph_index in range(pyg_batch.ptr.size(0) - 1):
        node_start, node_end = pyg_batch.ptr[graph_index], pyg_batch.ptr[graph_index + 1]
        node_indices = torch.arange(node_start, node_end)

        # Find edges that belong to the current graph
        edge_mask = (pyg_batch.batch[pyg_batch.edge_index[0]] == graph_index) & \
                    (pyg_batch.batch[pyg_batch.edge_index[1]] == graph_index)

        # Adjust edge_index to be local for each graph (subtract node_start to make node indices start from 0)
        edge_index = pyg_batch.edge_index[:, edge_mask] - node_start

        # Create a new Data object for the current graph
        graph = Data()

        # Dynamically assign the attributes in the batch to the graph
        for key, value in pyg_batch.items():
            if key in ['edge_index', 'batch', 'ptr']:
                continue  # Skip attributes that are not graph-specific

            # Handle edge-related attributes
            if value.size(0) == pyg_batch.edge_index.size(1):
                graph[key] = value[edge_mask]

            # Handle node-related attributes
            elif value.size(0) == pyg_batch.x.size(0):
                graph[key] = value[node_indices]

            # Handle graph-level attributes (single value per graph)
            elif value.size(0) == pyg_batch.ptr.size(0) - 1:
                graph[key] = value[graph_index].unsqueeze(0)

        # Set the adjusted edge_index for the graph
        graph.edge_index = edge_index

        graph_list.append(graph)

    return graph_list

# def extract_and_assign_edge_weights(pyg_batch, graph_list):
#     """
#     Extract edge weights for each graph in the batch and assign them back to each graph instance.
#
#     Args:
#         pyg_batch: Batched graph data from PyTorch Geometric
#         graph_list: List of individual graph instances (Data objects) from pyg_batch.to_data_list()
#
#     Returns:
#         Updated graph_list with edge_weight assigned to each instance.
#     """
#     for graph_index, instance in enumerate(graph_list):
#         # Extract the edge weights for the current graph
#         edge_indices = (pyg_batch.batch[pyg_batch.edge_index[0]] == graph_index)
#         instance.edge_weight = pyg_batch.edge_weight[edge_indices]
#
#     return graph_list