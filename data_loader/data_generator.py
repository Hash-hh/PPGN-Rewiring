import data_loader.data_helper as helper
import utils.config
import torch


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.batch_size = self.config.hyperparams.batch_size
        self.is_qm9 = self.config.dataset_name == 'QM9'
        self.is_ZINC = self.config.dataset_name == 'ZINC'
        self.labels_dtype = torch.float32 if self.is_qm9 else torch.long
        self.labels_dtype = torch.float32 if self.is_ZINC else torch.long

        self.load_data()

    # load the specified dataset in the config to the data_generator instance
    def load_data(self):
        if self.is_qm9:
            self.load_qm9_data()
        elif self.is_ZINC:
            self.load_ZINC_data()
        else:
            self.load_data_benchmark()

        self.split_val_test_to_batches()

    # load QM9 data set
    def load_qm9_data(self):
        (train_graphs, train_labels, train_pyg_list,
         val_graphs, val_labels, val_pyg_list,
         test_graphs, test_labels, test_pyg_list) = \
            helper.load_qm9(self.config.target_param, self.config.candidates, self.config.debug)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std  # Needed for postprocess, multiply mean abs distance by this std

        self.train_pyg_list = train_pyg_list
        self.val_pyg_list = val_pyg_list
        self.test_pyg_list = test_pyg_list

    # load ZINC data set
    def load_ZINC_data(self):
        (train_graphs, train_labels, train_pyg_list,
         val_graphs, val_labels, val_pyg_list,
         test_graphs, test_labels, test_pyg_list) = \
            helper.load_ZINC(self.config.target_param, self.config.candidates, self.config.debug, self.config.train_percent)

        # preprocess all labels by train set mean and std
        train_labels_mean = train_labels.mean(axis=0)
        train_labels_std = train_labels.std(axis=0)
        train_labels = (train_labels - train_labels_mean) / train_labels_std
        val_labels = (val_labels - train_labels_mean) / train_labels_std
        test_labels = (test_labels - train_labels_mean) / train_labels_std

        self.train_graphs, self.train_labels = train_graphs, train_labels
        self.val_graphs, self.val_labels = val_graphs, val_labels
        self.test_graphs, self.test_labels = test_graphs, test_labels

        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)
        self.test_size = len(self.test_graphs)
        self.labels_std = train_labels_std

        self.train_pyg_list = train_pyg_list
        self.val_pyg_list = val_pyg_list
        self.test_pyg_list = test_pyg_list

    # load data for a benchmark graph (COLLAB, NCI1, NCI109, MUTAG, PTC, IMDBBINARY, IMDBMULTI, PROTEINS)
    def load_data_benchmark(self):
        graphs, labels = helper.load_dataset(self.config.dataset_name)
        # if no fold specify creates random split to train and validation
        if self.config.num_fold is None:
            graphs, labels = helper.shuffle(graphs, labels)
            idx = len(graphs) // 10
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[idx:], labels[idx:], graphs[:idx], labels[:idx]
        elif self.config.num_fold == 0:
            train_idx, test_idx = helper.get_parameter_split(self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[
                train_idx], graphs[test_idx], labels[test_idx]
        else:
            train_idx, test_idx = helper.get_train_val_indexes(self.config.num_fold, self.config.dataset_name)
            self.train_graphs, self.train_labels, self.val_graphs, self.val_labels = graphs[train_idx], labels[train_idx], graphs[test_idx], labels[
                test_idx]
        # change validation graphs to the right shape
        self.train_size = len(self.train_graphs)
        self.val_size = len(self.val_graphs)

    def next_batch(self):
        graphs, labels, graphs_pyg = next(self.iter)
        graphs, labels = torch.cuda.FloatTensor(graphs), torch.tensor(labels, device='cuda', dtype=self.labels_dtype)
        graphs_pyg = [graph.to('cuda') for graph in graphs_pyg]
        return graphs, labels, graphs_pyg

    # initialize an iterator from the data for one training epoch
    def initialize(self, what_set):
        if what_set == 'train':
            self.reshuffle_data()
        elif what_set == 'val' or what_set == 'validation':
            self.iter = zip(self.val_graphs_batches, self.val_labels_batches, self.val_graphs_pyg_batches)
        elif what_set == 'test':
            self.iter = zip(self.test_graphs_batches, self.test_labels_batches, self.test_graphs_pyg_batches)
        else:
            raise ValueError("what_set should be either 'train', 'val' or 'test'")

    def reshuffle_data(self):
        """
        Reshuffle train data between epochs
        """
        graphs, labels, graphs_pyg = helper.group_same_size(self.train_graphs, self.train_labels, self.train_pyg_list)
        graphs, labels, graphs_pyg = helper.shuffle_same_size(graphs, labels, graphs_pyg)
        graphs, labels, graphs_pyg = helper.split_to_batches(graphs, labels, self.batch_size, graphs_pyg)
        self.num_iterations_train = len(graphs)
        graphs, labels, graphs_pyg = helper.shuffle(graphs, labels, graphs_pyg)
        self.iter = zip(graphs, labels, graphs_pyg)

    def split_val_test_to_batches(self):
        # Split the val and test sets to batchs, no shuffling is needed
        graphs, labels, graphs_pyg = helper.group_same_size(self.val_graphs, self.val_labels, self.val_pyg_list)  # Group same size graphs
        graphs, labels, graphs_pyg = helper.split_to_batches(graphs, labels, self.batch_size, graphs_pyg)  # Split batches for each size [graph_size, num_graphs, features(19), nodes, nodes]
        self.num_iterations_val = len(graphs)
        self.val_graphs_batches, self.val_labels_batches, self.val_graphs_pyg_batches = graphs, labels, graphs_pyg

        if self.is_qm9 or self.is_ZINC:
            # Benchmark graphs have no test sets
            graphs, labels, graphs_pyg = helper.group_same_size(self.test_graphs, self.test_labels, self.test_pyg_list)
            graphs, labels, graphs_pyg = helper.split_to_batches(graphs, labels, self.batch_size, graphs_pyg)
            self.num_iterations_test = len(graphs)
            self.test_graphs_batches, self.test_labels_batches, self.test_graphs_pyg_batches = graphs, labels, graphs_pyg


if __name__ == '__main__':
    config = utils.config.process_config('../configs/10fold_config.json')
    data = DataGenerator(config)
    data.initialize('train')


