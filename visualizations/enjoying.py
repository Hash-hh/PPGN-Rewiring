from models.hybrid_model import HybridModel
from utils.config import process_config
import os
import torch
from data_loader.data_generator import DataGenerator
from utils.pyg_to_tensor import manually_split_graphs_general
from utils_enjoying import visualize_rewired_graph

config = process_config('../configs/ZINC_config.json', dataset_name='ZINC')

# Instantiate the model
model = HybridModel(config)

# Load the saved model weights
load_dir = '../../experiments/ZINCZINC2024_10_03_22_24_10/checkpoint/'
# load_dir = '../../experiments/ZINCZINC2024_10_03_22_19_01/checkpoint/'
checkpoint = torch.load(os.path.join(load_dir, 'best.tar'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(torch.device('cuda'))
model.eval()  # Set model to evaluation mode

# Enable rewiring in the config (for visualization purposes)
model.config.return_rewiring = True

# Load data
data_loader = DataGenerator(config)
data_loader.initialize('test')
graphs, labels, graphs_pyg = data_loader.next_batch()


scores, repeated_labels, rewired_graph = model(graphs, labels, graphs_pyg, train=False)

rewired_graph_split = manually_split_graphs_general(rewired_graph[0])

# print(scores)
# print(repeated_labels)
# print(rewired_graph)


val_ensemble = model.config.sampling.val_ensemble

for i in range(5):
    visualize_rewired_graph(i, graphs_pyg, rewired_graph_split, ensemble_size=val_ensemble)
