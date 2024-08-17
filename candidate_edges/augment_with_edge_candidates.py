from typing import Optional, List

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (is_undirected,
                                   to_undirected,
                                   add_remaining_self_loops,
                                   coalesce)

class AugmentWithEdgeCandidate():
    def __init__(self, heuristic, num_candidate, directed):
        super(AugmentWithEdgeCandidate, self).__init__()
        self.heu = heuristic
        self.num_candidate = num_candidate
        self.directed = directed

    def __call__(self, graph: Data):
        # assert is_undirected(graph.edge_index, num_nodes=graph.num_nodes)
        edge_index = graph.edge_index.numpy()

        if self.heu == 'longest_path':
            mat = csr_matrix((np.ones(edge_index.shape[1]),
                              (edge_index[0], edge_index[1])),
                             shape=(graph.num_nodes, graph.num_nodes))

            mat = shortest_path(mat, directed=False, return_predecessors=False)
            mat[np.isinf(mat)] = -1.
            mat[mat == -1] = mat.max() + 1
        elif self.heu == 'node_similarity':
            x = graph.x
            if x.dtype == torch.long:
                x = torch.nn.functional.one_hot(x).reshape(x.shape[0], -1).float()
            x = torch.nn.functional.normalize(x, p=2.0, dim=1).numpy()
            mat = x @ x.T
            mat[np.arange(x.shape[0]), np.arange(x.shape[0])] = 0.
        elif self.heu == 'l1_similarity_per_node':
            x = graph.x
            assert torch.all(torch.unique(x) == torch.tensor([0, 1]))
            mat = - torch.linalg.norm(x[:, None, :] - x[None], ord=1, dim=2).numpy()
            mat[np.arange(x.shape[0]), np.arange(x.shape[0])] = -1.e8
        elif self.heu == 'all':
            mat = None
        else:
            raise NotImplementedError

        if not self.directed:
            candidate_idx = np.vstack(np.triu_indices(graph.num_nodes, k=1))
        else:
            candidate_idx = np.vstack(np.triu_indices(graph.num_nodes, k=-graph.num_nodes))
            candidate_idx = candidate_idx[:, candidate_idx[0] != candidate_idx[1]]  # no self loop

        # exclude original edges
        candidate_idx_id = candidate_idx[0] * graph.num_nodes + candidate_idx[1]
        org_edge_index_id = edge_index[0] * graph.num_nodes + edge_index[1]
        multi_hop_idx = np.logical_not(np.in1d(candidate_idx_id, org_edge_index_id))
        candidate_idx = candidate_idx[:, multi_hop_idx]

        if self.heu in ['node_similarity', 'longest_path', 'all']:
            # globally
            if self.heu in ['node_similarity', 'longest_path']:
                distances = mat[candidate_idx[0], candidate_idx[1]]
                edge_candidate = candidate_idx[:, np.argsort(distances)[-self.num_candidate:]]
            else:
                edge_candidate = candidate_idx
        elif self.heu in ['l1_similarity_per_node']:
            # per node
            if self.directed:
                raise NotImplementedError
            else:
                thresh = np.sort(mat, axis=1)[:, -self.num_candidate][:, None]
                mask = mat >= thresh
                thresh = np.sort(mat, axis=0)[-self.num_candidate][None]
                mask = np.logical_or(mat >= thresh, mask)
                mask = np.logical_or(mask.T, mask)

                edge_candidate = candidate_idx[:, mask[candidate_idx[0], candidate_idx[1]].nonzero()[0]]
        else:
            raise NotImplementedError

        graph.edge_candidate = torch.from_numpy(edge_candidate).T
        graph.num_edge_candidate = edge_candidate.shape[1]
        return graph
