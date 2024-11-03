from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
import torch
from candidate_edges.augment_with_edge_candidates import AugmentWithEdgeCandidate
from candidates_helper import convert_data_to_dict_zinc, ZINC_to_pyg_data_list
from joblib import dump
import pickle
import os
from tqdm import tqdm
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
aug = AugmentWithEdgeCandidate(heuristic='longest_path', num_candidate=100, directed=False)

def get_object_size(obj):
    temp_file = 'temp_size_calc.pickle'
    with open(temp_file, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(temp_file)
    os.remove(temp_file)
    return size

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
which_list = ['train', 'val', 'test']

for which_set in which_list:
    base_path = os.path.join(BASE_DIR, "data", "ZINC_candidates", f"ZINC_{which_set}.p")
    base_path_pyg = os.path.join(BASE_DIR, "data", "ZINC_candidates", f"ZINC_pyg_{which_set}.p")

    pyg_data_list_raw = ZINC(root='/data/ZINC', split=which_set, subset=True)
    pyg_data_list = ZINC_to_pyg_data_list(pyg_data_list_raw)  # preprocess the data (otherwise it will be too large)

    # Filter out graphs with fewer than `min_num_nodes` nodes
    min_num_nodes = 2  # 25
    max_num_nodes = 100  # 50

    # if which_set == 'train':  # only filter the training set
    #     filtered_pyg_data = [data for data in pyg_data_list if min_num_nodes <= data.x.size(0) <= max_num_nodes]
    # else:
    #     filtered_pyg_data = pyg_data_list

    filtered_pyg_data = [data for data in pyg_data_list if min_num_nodes <= data.x.size(0) <= max_num_nodes]


    # print(f"Processing {which_set} data...")
    filtered_pyg_data_list = filtered_pyg_data
    aug_pyg_data_list = [aug(data) for data in tqdm(filtered_pyg_data_list, desc=f"Processing {which_set} Data")]
    aug_pyg_data_batch = Batch.from_data_list(aug_pyg_data_list)
    dict_list = convert_data_to_dict_zinc(aug_pyg_data_list)

    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Save dict_list
    print(f"Saving {which_set} dict data...")
    with open(base_path, 'wb') as f:
        pickle.dump(dict_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # dict_size = get_object_size(dict_list)
    # print(f"Size of {which_set} dict_list: {dict_size / (1024 ** 2):.2f} MB")
    # print(f"Saved {which_set} dict data to {base_path}")

    # Save PyG data
    # print(f"Saving {which_set} PyG data...")
    # with open(base_path_pyg, 'wb') as f:
    #     pickle.dump(aug_pyg_data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    torch.save(aug_pyg_data_batch, base_path_pyg)

    # pyg_size = get_object_size(aug_pyg_data_list)
    # print(f"Size of {which_set} PyG data: {pyg_size / (1024 ** 2):.2f} MB")
    # print(f"Saved {which_set} PyG data to {base_path_pyg}")

print("Done!")