import torch


def transform_pyg_batch(pyg_batch, num_features=20):

    assert len(pyg_batch) == 1, "We are only handling the case with one batch of graphs."
    pyg_batch = pyg_batch[0]
    graph_list = pyg_batch.to_data_list()

    transformed_graphs = [transform_graph(graph, num_features=num_features) for graph in graph_list]

    batched_features = torch.stack(transformed_graphs, dim=0)

    return batched_features


def transform_graph(data, num_features=20):
    num_nodes = data.num_nodes
    graph_features = torch.zeros(num_features, num_nodes, num_nodes, device=data.x.device, dtype=data.x.dtype)

    # Efficiently create diagonal matrices for node features
    diags = torch.diag_embed(data.x.T)  # Embedding the first 13 features
    graph_features[:13] = diags

    # Additional features
    if hasattr(data, 'distance_mat'):
        graph_features[13] = data.distance_mat
    if hasattr(data, 'affinity'):
        graph_features[14] = data.affinity
    if hasattr(data, 'edge_features'):
        graph_features[15:19] = data.edge_features.view(4, num_nodes, num_nodes)  # Adjust based on structure
    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            graph_features[20] = data.edge_weight

    return graph_features
