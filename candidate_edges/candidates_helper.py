import torch
import os
import pickle
from torch_geometric.data import InMemoryDataset, Data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convert_data_to_dict(data_list):
    dict_list = []

    for instance in data_list:
        # Convert each attribute of the Data object to the desired format
        x = instance.x
        distance_mat = instance.distance_mat
        affinity = instance.affinity
        edge_features = instance.edge_features
        edge_index = instance.edge_index
        y = instance.y
        edge_candidate = instance.edge_candidate
        num_edge_candidate = instance.num_edge_candidate
        pos = instance.pos
        edge_attr = instance.edge_attr

        molecule_dict = {
            'usable_features': {
                'x': x,
                'distance_mat': distance_mat,
                'affinity': affinity,
                'edge_features': edge_features
            },
            'original_features': {
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'pos': pos
            },
            'y': y,
            'candidates': {
                'edge_candidates': edge_candidate,
                'num_edge_candidate': num_edge_candidate
            }

        }

        dict_list.append(molecule_dict)

    return dict_list


def convert_qm9_to_pyg_data_list(which_set, num_samples=None):
    """
    Maps QM9 dataset to PyG Data object
    :param which_set: str, 'train', 'valid', 'test'
    :return: PyG Data object
    """

    base_path = BASE_DIR + "/data/QM9/QM9_{}.p".format(which_set)

    data_list = []

    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in data:
            # Extract features for the current molecule
            x = torch.tensor(instance['usable_features']['x'], dtype=torch.float)  # Node features
            distance_mat = torch.tensor(instance['usable_features']['distance_mat'], dtype=torch.float)
            affinity = torch.tensor(instance['usable_features']['affinity'], dtype=torch.float)
            edge_features = torch.tensor(instance['usable_features']['edge_features'], dtype=torch.float)
            edge_index = torch.tensor(instance['original_features']['edge_index'])
            pos = torch.tensor(instance['original_features']['pos'], dtype=torch.float)
            edge_attr = torch.tensor(instance['original_features']['edge_attr'], dtype=torch.float)
            y = torch.tensor(instance['y'], dtype=torch.float)  # Labels

            # Create a Data object
            data = Data(x=x, edge_index=edge_index,
                        distance_mat=distance_mat,
                        affinity=affinity,
                        edge_features=edge_features,
                        edge_attr=edge_attr,
                        pos=pos,
                        y=y)

            data_list.append(data)


        if num_samples is not None:
            data_list = data_list[:num_samples]

    return data_list