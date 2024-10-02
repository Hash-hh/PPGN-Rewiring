import torch


def transform_pyg_batch(repeated_data, pyg_batch, num_features=20, is_QM9=False, is_ZINC=False):

    assert len(pyg_batch) == 1, "We are only handling the case with one batch of graphs."
    pyg_batch = pyg_batch[0]
    graph_list = pyg_batch.to_data_list()

    transformed_graphs = [transform_graph(instance, pyg_graph, num_features=num_features, is_QM9=is_QM9, is_ZINC=is_ZINC)
                          for instance, pyg_graph in zip(repeated_data, graph_list)]

    batched_features = torch.stack(transformed_graphs, dim=0)

    return batched_features


def transform_graph(instance, pyg_graph, num_features=20, is_QM9=False, is_ZINC=False):
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
        if hasattr(pyg_graph, 'edge_weight'):
            if pyg_graph.edge_weight is not None:
                graph_features[20] = pyg_graph.edge_weight

    elif is_ZINC:
        graph_features[:21] = diags

        if hasattr(pyg_graph, 'edge_features'):
            graph_features[21:25] = pyg_graph.edge_features.view(4, num_nodes, num_nodes)
        else:
            graph_features[21:25] = instance[-4:, :, :]  # Use the last 4 features from the instance

        if hasattr(pyg_graph, 'edge_weight'):
            if pyg_graph.edge_weight is not None:
                graph_features[25] = pyg_graph.edge_weight


    return graph_features
