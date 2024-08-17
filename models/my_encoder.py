import torch
from ml_collections import ConfigDict
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred.mol_encoder import AtomEncoder

from torch import nn as nn

from torch_geometric.nn import MLP

class FeatureEncoder(torch.nn.Module):

    def __init__(self,
                 dim_in,
                 hidden,
                 type_encoder='linear',
                 lap_encoder=None,
                 rw_encoder=None):
        super(FeatureEncoder, self).__init__()

        lin_hidden = hidden
        # if lap_encoder is not None:
        #     lin_hidden -= lap_encoder.dim_pe
        # if rw_encoder is not None:
        #     lin_hidden -= rw_encoder.dim_pe

        if type_encoder == 'linear':
            self.linear_embed = nn.Linear(dim_in, lin_hidden)
        # elif type_encoder == 'bi_embedding':
        #     self.linear_embed = BiEmbedding(dim_in, lin_hidden)
        # elif type_encoder == 'bi_embedding_cat':
        #     assert lin_hidden % 2 == 0, 'lin_hidden must be even'
        #     # n_features hardcoded right now
        #     self.linear_embed = BiEmbedding_cat(n_nodes=dim_in, n_features=2, hidden=lin_hidden//2)
        # elif type_encoder == 'atomencoder':
        #     self.linear_embed = AtomEncoder(lin_hidden)
        # elif type_encoder == 'embedding':
        #     # https://github.com/rampasek/GraphGPS/blob/28015707cbab7f8ad72bed0ee872d068ea59c94b/graphgps/encoder/type_dict_encoder.py#L82
        #     raise NotImplementedError
        else:
            raise ValueError

        # if lap_encoder is not None:
        #     self.lap_encoder = LapPENodeEncoder(hidden,
        #                                         hidden - rw_encoder.dim_pe if rw_encoder is not None else hidden,
        #                                         lap_encoder,
        #                                         expand_x=False)
        # else:
        #     self.lap_encoder = None
        #
        # if rw_encoder is not None:
        #     self.rw_encoder = RWSENodeEncoder(hidden, hidden, rw_encoder, expand_x=False)
        # else:
        #     self.rw_encoder = None

    def forward(self, batch):
        x = self.linear_embed(batch.x)
        # if self.lap_encoder is not None:
        #     x = self.lap_encoder(x, batch)
        # if self.rw_encoder is not None:
        #     x = self.rw_encoder(x, batch)
        return x


