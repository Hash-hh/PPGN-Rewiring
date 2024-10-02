import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Batch

NUM_LABELS = {'ENZYMES': 3, 'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38,
              'PROTEINS': 3, 'PTC': 22, 'DD': 89}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(ds_name):
    """
    construct graphs and labels from dataset text in data folder
    :param ds_name: name of data set you want to load
    :return: two numpy arrays of shape (num_of_graphs).
            the graphs array contains in each entry a ndarray represent adjacency matrix of a graph of shape (num_vertex, num_vertex, num_vertex_labels)
            the labels array in index i represent the class of graphs[i]
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/{0}.txt".format(ds_name)
    graphs = []
    labels = []
    with open(directory, "r") as data:
        num_graphs = int(data.readline().rstrip().split(" ")[0])
        for i in range(num_graphs):
            graph_meta = data.readline().rstrip().split(" ")
            num_vertex = int(graph_meta[0])
            curr_graph = np.zeros(shape=(num_vertex, num_vertex, NUM_LABELS[ds_name]+1), dtype=np.float32)
            labels.append(int(graph_meta[1]))
            for j in range(num_vertex):
                vertex = data.readline().rstrip().split(" ")
                if NUM_LABELS[ds_name] != 0:
                    curr_graph[j, j, int(vertex[0])+1]= 1.
                for k in range(2,len(vertex)):
                    curr_graph[j, int(vertex[k]), 0] = 1.
            curr_graph = normalize_graph(curr_graph)
            graphs.append(curr_graph)
    graphs = np.array(graphs, dtype="object")
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2,0,1])
    return graphs, np.array(labels)


def load_qm9(target_param, candidates, debug):
    """
    Constructs the graphs and labels of QM9 data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 12, (or Nx1 is target_param is not False)
                 val_graphs: N_val,
                 val_labels: N_train x 12, (or Nx1 is target_param is not False)
                 test_graphs: N_test,
                 test_labels: N_test x 12, (or Nx1 is target_param is not False)
                 each graph of shape: 19 x Nodes x Nodes (CHW representation)

    update: added return of PyG data list for each set
    """
    train_graphs, train_labels, train_pyg_list = load_qm9_aux('train', target_param, candidates)
    val_graphs, val_labels, val_pyg_list = load_qm9_aux('val', target_param, candidates)
    test_graphs, test_labels, test_pyg_list = load_qm9_aux('test', target_param, candidates)

    if debug:  # Return only the first 10 graphs for debugging
        return (train_graphs[:10], train_labels[:10], train_pyg_list[:10],
                val_graphs[:10], val_labels[:10], val_pyg_list[:10],
                test_graphs[:10], test_labels[:10], test_pyg_list[:10])

    return (train_graphs, train_labels, train_pyg_list,
            val_graphs, val_labels, val_pyg_list,
            test_graphs, test_labels, test_pyg_list)


def load_qm9_aux(which_set, target_param, candidate_edges=False):
    """
    Read and construct the graphs and labels of QM9 data set, already split to train, val and test sets
    :param which_set: 'test', 'train' or 'val'
    :param target_param: if not false, return the labels for this specific param only
    :param candidate_edges: if True, return the PyG data list as well
    :return: graphs: (N,)
             labels: N x 12, (or Nx1 is target_param is not False)
             each graph of shape: 19 x Nodes x Nodes (CHW representation)
    """
    base_path = BASE_DIR + "/data/QM9_candidates/QM9_{}.p".format(which_set)
    graphs, labels = [], []
    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in data:
            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 19))
            for i in range(13):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 13] = instance['usable_features']['distance_mat']
            graph[:, :, 14] = instance['usable_features']['affinity']
            graph[:, :, 15:] = instance['usable_features']['edge_features']  # shape n x n x 4
            graphs.append(graph)
    graphs = np.array(graphs, dtype="object")
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # [nodes_num, nodes_num, features] -> [features, nodes_num, nodes_num]
    labels = np.array(labels).squeeze()  # shape N x 12  - remove the initial singleton dimension
    if target_param is not False:  # regression over a specific target, not all 12 elements
        labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1

    if candidate_edges:
        base_path_pyg = BASE_DIR + "/data/QM9_candidates/QM9_pyg_{}.p".format(which_set)

        pyg_graphs = []
        # Returning the PyG data list as well
        with open(base_path_pyg, 'rb') as f:
            pyg_data_list = pickle.load(f)
            for data in pyg_data_list:
                pyg_graphs.append(data)

        return graphs, labels, pyg_graphs

    return graphs, labels, None


def load_ZINC(target_param, candidates, debug):
    """
    Constructs the graphs and labels of ZINC data set, already split to train, val and test sets
    :return: 6 numpy arrays:
                 train_graphs: N_train,
                 train_labels: N_train x 1
                 val_graphs: N_val,
                 val_labels: N_train x 1
                 test_graphs: N_test,
                 test_labels: N_test x 1
                 each graph of shape: 25 x Nodes x Nodes (CHW representation)

    update: added return of PyG data list for each set
    """
    train_graphs, train_labels, train_pyg_list = load_ZINC_aux('train', target_param, candidates)
    val_graphs, val_labels, val_pyg_list = load_ZINC_aux('val', target_param, candidates)
    test_graphs, test_labels, test_pyg_list = load_ZINC_aux('test', target_param, candidates)

    if debug:  # Return only the first 10 graphs for debugging
        return (train_graphs[:10], train_labels[:10], train_pyg_list[:10],
                val_graphs[:10], val_labels[:10], val_pyg_list[:10],
                test_graphs[:10], test_labels[:10], test_pyg_list[:10])

    return (train_graphs, train_labels, train_pyg_list,
            val_graphs, val_labels, val_pyg_list,
            test_graphs, test_labels, test_pyg_list)


def load_ZINC_aux(which_set, target_param, candidate_edges=False):
    """
    Read and construct the graphs and labels of ZINC data set, already split to train, val and test sets
    :param which_set: 'test', 'train' or 'val'
    :param target_param: if not false, return the labels for this specific param only
    :param candidate_edges: if True, return the PyG data list as well
    :return: graphs: (N,)
             labels: N x 1
             each graph of shape: 25 x Nodes x Nodes (CHW representation)
    """
    base_path = BASE_DIR + "/data/ZINC_candidates/ZINC_{}.p".format(which_set)
    graphs, labels = [], []
    with open(base_path, 'rb') as f:
        data = pickle.load(f)
        for instance in data:
            labels.append(instance['y'])
            nodes_num = instance['usable_features']['x'].shape[0]
            graph = np.empty((nodes_num, nodes_num, 21 + 4))
            for i in range(21):
                # 13 features per node - for each, create a diag matrix of it as a feature
                graph[:, :, i] = np.diag(instance['usable_features']['x'][:, i])
            graph[:, :, 21:] = instance['usable_features']['edge_features']  # shape n x n x 4
            graphs.append(graph)
    graphs = np.array(graphs, dtype="object")
    for i in range(graphs.shape[0]):
        graphs[i] = np.transpose(graphs[i], [2, 0, 1])  # [nodes_num, nodes_num, features] -> [features, nodes_num, nodes_num]
    labels = np.array(labels)  #.squeeze()  # shape N x 12  - remove the initial singleton dimension
    if target_param is not False:  # regression over a specific target, not all 12 elements
        labels = labels[:, target_param].reshape(-1, 1)  # shape N x 1

    if candidate_edges:
        base_path_pyg = BASE_DIR + "/data/ZINC_candidates/ZINC_pyg_{}.p".format(which_set)

        pyg_graphs = []
        # Returning the PyG data list as well
        pyg_data_list = torch.load(base_path_pyg, weights_only=False)
        # Check if the data is in a batch
        if isinstance(pyg_data_list, Batch):
            # Convert the batch to a list of individual graphs
            pyg_graphs = pyg_data_list.to_data_list()
        else:
            # If not a batch, directly append individual graphs
            for data in pyg_data_list:
                pyg_graphs.append(data)

        return graphs, labels, pyg_graphs

    return graphs, labels, None


def get_train_val_indexes(num_val, ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param num_val: number of the split
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/10fold_idx".format(ds_name)
    train_file = "train_idx-{0}.txt".format(num_val)
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "test_idx-{0}.txt".format(num_val)
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def get_parameter_split(ds_name):
    """
    reads the indexes of a specific split to train and validation sets from data folder
    :param ds_name: name of data set
    :return: indexes of the train and test graphs
    """
    directory = BASE_DIR + "/data/benchmark_graphs/{0}/".format(ds_name)
    train_file = "tests_train_split.txt"
    train_idx=[]
    with open(os.path.join(directory, train_file), 'r') as file:
        for line in file:
            train_idx.append(int(line.rstrip()))
    test_file = "tests_val_split.txt"
    test_idx = []
    with open(os.path.join(directory, test_file), 'r') as file:
        for line in file:
            test_idx.append(int(line.rstrip()))
    return train_idx, test_idx


def group_same_size(graphs, labels, graph_pyg=None):
    """
    Group graphs of the same size into the same array.

    :param graphs: numpy array of shape (num_of_graphs) of numpy arrays of graphs' adjacency matrix
    :param labels: numpy array of labels
    :param graph_pyg: list of graph objects in PyTorch Geometric format, or None
    :return: three lists:
             - r_graphs: list of numpy arrays of graphs grouped by size
             - r_labels: list of numpy arrays of labels grouped by graph size
             - r_graph_pyg: list of lists of PyTorch Geometric graph objects grouped by size or None if graph_pyg is None
    """
    sizes = list(map(lambda t: t.shape[1], graphs))  # Size based on the number of vertices in adjacency matrices
    indexes = np.argsort(sizes)
    graphs = graphs[indexes]
    labels = labels[indexes]

    r_graphs = []
    r_labels = []
    r_graph_pyg = [] if graph_pyg is not None else None

    if graph_pyg is not None:
        graph_pyg = [graph_pyg[i] for i in indexes]  # Reorder graph_pyg according to the sorted indexes

    one_size = []
    one_pyg = [] if graph_pyg is not None else None
    start = 0
    size = graphs[0].shape[1]  # Smallest graph by number of vertices

    for i in range(len(graphs)):
        if graphs[i].shape[1] == size:
            one_size.append(np.expand_dims(graphs[i], axis=0))
            if graph_pyg is not None:
                one_pyg.append(graph_pyg[i])
        else:
            r_graphs.append(np.concatenate(one_size, axis=0))
            r_labels.append(np.array(labels[start:i]))
            if graph_pyg is not None:
                r_graph_pyg.append(one_pyg)
                one_pyg = [graph_pyg[i]]
            start = i
            one_size = [np.expand_dims(graphs[i], axis=0)]
            size = graphs[i].shape[1]

    r_graphs.append(np.concatenate(one_size, axis=0))
    r_labels.append(np.array(labels[start:]))
    if graph_pyg is not None:
        r_graph_pyg.append(one_pyg)

    return r_graphs, r_labels, r_graph_pyg


# helper method to shuffle each same size graphs array
def shuffle_same_size(graphs, labels, graph_pyg=None):
    """
    Shuffle graphs, labels, and optionally PyTorch Geometric graphs of the same size while maintaining correspondence.

    :param graphs: list of numpy arrays of graphs' adjacency matrices
    :param labels: list of numpy arrays of labels
    :param graph_pyg: list of lists of PyTorch Geometric graph objects, or None
    :return: shuffled versions of the input arrays
    """
    r_graphs, r_labels, r_graph_pyg = [], [], [] if graph_pyg is not None else None

    for i in range(len(labels)):
        if graph_pyg is not None:
            curr_graph, curr_labels, curr_pyg = shuffle(graphs[i], labels[i], graph_pyg[i])
            r_graph_pyg.append(curr_pyg)
        else:
            curr_graph, curr_labels, _ = shuffle(graphs[i], labels[i], None)

        r_graphs.append(curr_graph)
        r_labels.append(curr_labels)

    return r_graphs, r_labels, r_graph_pyg


def split_to_batches(graphs, labels, size, graph_pyg):
    """
    Split the same size graphs array into batches of specified size.
    The last batch will have the size of num_of_graphs_this_size % size.

    :param graphs: array of arrays of same size graphs
    :param labels: the corresponding labels of the graphs
    :param graph_pyg: list of lists of PyTorch Geometric graph objects
    :param size: batch size
    :return: three arrays:
             - graphs array of arrays in size (batch, num vertex, num vertex. num vertex labels)
             - corresponds labels
             - corresponding graph_pyg
    """
    r_graphs = []
    r_labels = []
    r_graph_pyg = [] if graph_pyg is not None else None

    for k in range(len(graphs)):
        num_batches = (len(graphs[k]) + size - 1) // size  # Calculates the number of batches
        for i in range(num_batches):
            start_idx = i * size
            end_idx = min((i + 1) * size, len(graphs[k]))

            r_graphs.append(graphs[k][start_idx:end_idx])
            r_labels.append(labels[k][start_idx:end_idx])
            if graph_pyg is not None:
                r_graph_pyg.append(graph_pyg[k][start_idx:end_idx])

    # Create arrays of objects to avoid potential numpy dtype issues
    ret1, ret2, ret3 = (np.array(r_graphs, dtype=object),
                        np.array(r_labels, dtype=object),
                        r_graph_pyg if graph_pyg is not None else None)
                        # np.array(r_graph_pyg, dtype=object) if graph_pyg is not None else None)

    return ret1, ret2, ret3


def shuffle(graphs, labels, pyg_graphs):
    """
    Helper method to shuffle the same way graphs, labels, and PyTorch Geometric graphs arrays.
    """
    shf = np.arange(labels.shape[0], dtype=np.int32)
    np.random.shuffle(shf)
    shuffled_graphs = np.array(graphs)[shf]
    shuffled_labels = labels[shf]
    shuffled_pyg_graphs = [pyg_graphs[i] for i in shf] if pyg_graphs is not None else None
    return shuffled_graphs, shuffled_labels, shuffled_pyg_graphs



def normalize_graph(curr_graph):

    split = np.split(curr_graph, [1], axis=2)

    adj = np.squeeze(split[0], axis=2)
    deg = np.sqrt(np.sum(adj, 0))
    deg = np.divide(1., deg, out=np.zeros_like(deg), where=deg!=0)
    normal = np.diag(deg)
    norm_adj = np.expand_dims(np.matmul(np.matmul(normal, adj), normal), axis=2)
    ones = np.ones(shape=(curr_graph.shape[0], curr_graph.shape[1], curr_graph.shape[2]), dtype=np.float32)
    spred_adj = np.multiply(ones, norm_adj)
    labels= np.append(np.zeros(shape=(curr_graph.shape[0], curr_graph.shape[1], 1)), split[1], axis=2)
    return np.add(spred_adj, labels)


if __name__ == '__main__':
    graphs, labels = load_dataset("MUTAG")
    a, b = get_train_val_indexes(1, "MUTAG")
    print(np.transpose(graphs[a[0]], [1, 2, 0])[0])
