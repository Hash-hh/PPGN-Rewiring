import torch
import os
import pickle
from torch_geometric.data import InMemoryDataset, Data, Batch
import torch.nn.functional as F
from tqdm import tqdm



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


def convert_data_to_dict_zinc(data_list):
    dict_list = []

    for instance in tqdm(data_list, desc="Converting Data", unit="instance"):
        # Convert each attribute of the Data object to the desired format

        # Assign other features without modification
        y = instance.y
        edge_candidate = instance.edge_candidate
        num_edge_candidate = instance.num_edge_candidate

        # creating 3D adjacency matrix for edge representation
        num_nodes = instance.x.size(0)
        num_edge_types = 4  # Number of distinct edge types
        edge_feature_matrix = torch.zeros((num_nodes, num_nodes, num_edge_types))
        # Fill in the adjacency matrix with edge features
        edge_index = instance.edge_index
        edge_attr = instance.edge_attr
        for i in range(edge_index.size(1)):
            start, end = edge_index[:, i]
            # Directly assign the one-hot encoded edge attributes
            edge_feature_matrix[start, end] = edge_attr[i]

        # Create a dictionary for the molecule's data
        molecule_dict = {
            'usable_features': {
                'x': instance.x,  # Use one-hot encoded x
                'edge_features': edge_feature_matrix
            },
            # 'original_features': {
            #     'edge_index': edge_index,
            #     'x_original': atom_types,
            #     'edge_attr': edge_attr
            # },
            'y': y,
            'candidates': {
                'edge_candidates': edge_candidate,
                'num_edge_candidate': num_edge_candidate
            }
        }

        dict_list.append(molecule_dict)

    return dict_list


def ZINC_to_pyg_data_list(pyg_data_list, num_samples=None):
    """
    Maps ZINC dataset to PyG Data object
    :return: PyG Data object
    """
    data_list = []


    for instance in pyg_data_list:

        # Convert atom features (x) to one-hot encoding
        atom_types = instance.x.squeeze()  # Remove extra dimension if necessary
        num_atom_types = 21  # Number of distinct atom types (28 for full set)
        one_hot_x = F.one_hot(atom_types, num_classes=num_atom_types)  # One-hot encode and convert to float

        # Convert edge_attr (bond types) to one-hot encoding
        edge_attr = instance.edge_attr.squeeze()  # Remove extra dimension if necessary
        num_edge_types = 4  # Number of distinct edge types
        one_hot_edge_attr = F.one_hot(edge_attr, num_classes=num_edge_types)  # One-hot encode and convert to float


        y = instance.y

        # Create a Data object
        data = Data(x=one_hot_x, edge_index=instance.edge_index,
                    edge_attr=one_hot_edge_attr,
                    y=y)

        data_list.append(data)


    if num_samples is not None:
        data_list = data_list[:num_samples]

    return data_list
