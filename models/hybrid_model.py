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
        encoder = FeatureEncoder(dim_in=13, hidden=128)
        edge_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128))
        self.config = config
        self.upstream = EdgeSelector(encoder=encoder, edge_encoder=edge_encoder,
                                     in_dim=128,
                                     hid_size=128,
                                     gnn_layer=3,
                                     mlp_layer=1,
                                     use_deletion_head=True
                                     ).cuda()

        simple_sampler = EdgeSIMPLEBatched()

        self.rewiring = GraphRewirer(add_k=config.sampling.add_k,
                            del_k=config.sampling.del_k,
                            train_ensemble=config.sampling.train_ensemble,
                            val_ensemble=config.sampling.val_ensemble,
                            sampler=simple_sampler)

        self.downstream = Downstream(config).cuda()
        # self.basemodel = BaseModel(config).cuda()

    def forward(self, data, labels, graphs_pyg):

        repeated_data = data.repeat(3, 1, 1, 1)

        graphs_pyg_batch = Batch.from_data_list(graphs_pyg)

        select_edge_candidates, delete_edge_candidates, edge_candidate_idx = self.upstream(graphs_pyg_batch)
        new_data = self.rewiring(graphs_pyg_batch,
                                          select_edge_candidates,  # addition logits
                                          delete_edge_candidates,  # deletion logits
                                          edge_candidate_idx)  # candidate edge index (wholesale)

        new_data = transform_pyg_batch(new_data, num_features=20)


        scores = self.downstream(repeated_data, new_data)

        repeated_labels = labels.repeat(3,1)

        return scores, repeated_labels

        # old network (for testing)
        # return self.basemodel(data), labels
