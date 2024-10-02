from torch_geometric.datasets import ZINC
from candidate_edges.augment_with_edge_candidates import AugmentWithEdgeCandidate

dataset = ZINC(root='/data/ZINC', split='train', subset=True)

# Find the unique atom types and edge types across the entire dataset
unique_atom_types = set()
unique_edge_types = set()

# Iterate through the dataset to find unique atom types and edge types
for molecule in dataset:
    # Add unique atom types
    unique_atom_types.update(molecule.x.squeeze().tolist())

    # Add unique edge types
    unique_edge_types.update(molecule.edge_attr.squeeze().tolist())

# Maximum number of distinct atom and edge types
max_atom_type = max(unique_atom_types) + 1  # Adding 1 for one-hot encoding
max_edge_type = max(unique_edge_types) + 1  # Adding 1 for one-hot encoding

print(f"Maximum number of unique atom types: {max_atom_type}")
print(f"Maximum number of unique edge types: {max_edge_type}")


# # only keep molecules with more than n atoms
# dataset = [molecule for molecule in dataset if molecule.x.shape[0] > 25]
# print("Total number of graphs in the dataset after filtering: ", len(dataset))
#
# # add candidate edges
# aug = AugmentWithEdgeCandidate(heuristic='longest_path', num_candidate=100, directed=False)
# dataset = dataset[:100]
# dataset = [aug(molecule) for molecule in dataset]
#
# print("Total number of graphs in the dataset: ", len(dataset))




