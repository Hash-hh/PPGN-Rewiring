from candidate_edges.augment_with_edge_candidates import AugmentWithEdgeCandidate
from candidates_helper import convert_data_to_dict, convert_qm9_to_pyg_data_list
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

aug = AugmentWithEdgeCandidate(heuristic='longest_path', num_candidate=100, directed=False)

which_list = ['train', 'val', 'test']
for which_set in which_list:
    base_path = BASE_DIR + "/data/QM9_candidates/QM9_{}.p".format(which_set)
    base_path_pyg = BASE_DIR + "/data/QM9_candidates/QM9_pyg_{}.p".format(which_set)

    pyg_data_list = convert_qm9_to_pyg_data_list(which_set, num_samples=None)  # num_samples=None means all samples (put 5000 for testing)

    # Filter out graphs with fewer than `min_num_nodes` nodes
    min_num_nodes = 5  # remove graphs with fewer than min_num_nodes nodes
    filtered_pyg_data_list = [data for data in pyg_data_list if data.x.size(0) >= min_num_nodes]

    aug_pyg_data_list = [aug(data) for data in filtered_pyg_data_list]
    dict_list = convert_data_to_dict(aug_pyg_data_list)  # Convert PyG Data object to dictionary for PPGNN

    if not os.path.exists(os.path.dirname(base_path)):
        os.makedirs(os.path.dirname(base_path))
    with open(base_path, 'wb') as f:
        pickle.dump(dict_list, f)
        print(f"Saved {which_set} data to {base_path}")
    with open(base_path_pyg, 'wb') as f:
        pickle.dump(aug_pyg_data_list, f)
        print(f"Saved {which_set} data to {base_path_pyg}")

print("Done!")
