from itertools import islice
from typing import Callable

import torch
from models.base_model import BaseModel
from models.downstream import Downstream
from models.upstream import EdgeSelector
from models.my_encoder import FeatureEncoder
import torch.nn as nn
from torch_geometric.data import Batch
from simple.simple_scheme import EdgeSIMPLEBatched
from models.rewirer import GraphRewirer
from utils.pyg_to_tensor import transform_pyg_batch

class HybridModel(torch.nn.Module):
    def __init__(self, config):
        super(HybridModel, self).__init__()

        self.config = config
        self.is_QM9 = config.dataset_name == 'QM9'
        self.is_ZINC = config.dataset_name == 'ZINC'

        if self.is_QM9:
            dim_in = 13
        elif self.is_ZINC:
            dim_in = 21
        else:
            dim_in = 13
            print("Unknown dataset, using default dim_in=13")

        if config.rewiring:
            encoder = FeatureEncoder(dim_in=dim_in, hidden=32)  # 13 in the paper and 128 hidden
            edge_encoder = nn.Sequential(
                nn.Linear(4, 32),  # 4 in and 128 out in the paper
                nn.ReLU(),
                nn.Linear(32, 32))  # 128 in and 128 out in the paper
            self.config = config
            self.upstream = EdgeSelector(encoder=encoder, edge_encoder=edge_encoder,
                                         in_dim=64,  # 128 in the paper  # 64
                                         hid_size=64,  # 128 in the paper  # 64
                                         gnn_layer=2,  # 3 in the paper  # 2
                                         mlp_layer=1,
                                         use_deletion_head=True,
                                         ensemble=config.sampling.ensemble,
                                         ).cuda()

            simple_sampler = EdgeSIMPLEBatched()

            self.rewiring = GraphRewirer(add_k=config.sampling.add_k,
                                del_k=config.sampling.del_k,
                                train_ensemble=config.sampling.train_ensemble,
                                val_ensemble=config.sampling.val_ensemble,
                                sampler=simple_sampler)

            self.downstream = Downstream(config).cuda()

        else:
            self.basemodel = BaseModel(config).cuda()

    def forward(self, data, labels, graphs_pyg, train=True):

        num_features = 20 if self.is_QM9 else 26

        if self.config.rewiring:
            rewring_samples = self.config.sampling.train_ensemble if train else self.config.sampling.val_ensemble
            repeated_data = data.repeat(rewring_samples, 1, 1, 1)

            graphs_pyg_batch = Batch.from_data_list(graphs_pyg)
            select_edge_candidates, delete_edge_candidates, edge_candidate_idx = self.upstream(graphs_pyg_batch)
            rewired_graph = self.rewiring(graphs_pyg_batch,
                                              select_edge_candidates,  # addition logits
                                              delete_edge_candidates,  # deletion logits
                                              edge_candidate_idx)  # candidate edge index (wholesale)
            new_data = transform_pyg_batch(repeated_data, rewired_graph, num_features=num_features, is_QM9=self.is_QM9, is_ZINC=self.is_ZINC)
            scores = self.downstream(repeated_data, new_data)
            repeated_labels = labels.repeat(rewring_samples, 1)

            if self.config.return_rewiring:  # for visualization of rewiring
                return scores, repeated_labels, rewired_graph

            return scores, repeated_labels

        else:
            return self.basemodel(data), labels
