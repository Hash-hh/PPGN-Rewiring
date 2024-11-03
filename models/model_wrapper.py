import os
import models
import wandb
import torch
import torch.nn.functional as F
from models.base_model import BaseModel
from models.hybrid_model import HybridModel


class ModelWrapper(object):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        # self.model = BaseModel(config).cuda()  # TODO: Use hybrid model here
        self.model = HybridModel(config).cuda()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, best: bool, epoch: int, optimizer: torch.optim.Optimizer):
        filename = 'best.tar' if best else 'last.tar'
        print("Saving model as {}...".format(filename), end=' ')
        save_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   save_path)
        print("Model saved.")

        # Save model to WandB
        # if self.config.get('wandb', False):
        #     wandb.save(save_path)  # Upload the saved file to WandB
        #     print("Model also saved in WandB.")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, best: bool):
        """
        :param best: boolean, to load best model or last checkpoint
        :return: tuple of optimizer_state_dict, epoch
        """
        # self.model = models.base_model.BaseModel(self.config).to('cuda')
        filename = 'best.tar' if best else 'last.tar'
        print("Loading {}...".format(filename), end=' ')
        checkpoint = torch.load(os.path.join(self.config.checkpoint_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(torch.device('cuda'))
        print("Model loaded.")

        return checkpoint['optimizer_state_dict'], checkpoint['epoch']

    def loss_and_results(self, scores, labels):
        """
        :param scores: shape NxC
        :param labels: shape Nx1 for classification, shape NxC for regression (QM9)
        :return: tuple of (loss tensor, dists numpy array) for QM9
                          (loss tensor, number of correct predictions) for classification graphs
        """
        if self.config.dataset_name == 'QM9' or self.config.dataset_name == 'ZINC':
            differences = (scores-labels).abs().sum(dim=0)
            loss = differences.sum()
            dists = differences.detach().cpu().numpy()
            return loss, dists
        else:
            loss = F.cross_entropy(scores, labels, reduction='sum')
            correct_predictions = torch.eq(torch.argmax(scores, dim=1), labels).sum().cpu().item()
            return loss, correct_predictions

    def run_model_get_loss_and_results(self, input, labels, graphs_pyg, train=True):
        # scores = self.model(input)  # for when only using base model
        if self.config.return_rewiring:
            scores, repeated_labels, rewired_graph = self.model(input, labels, graphs_pyg, train)
            return self.loss_and_results(scores, repeated_labels)
        else:
            scores, labels = self.model(input, labels, graphs_pyg, train)  # feeding labels to get labels * train_ensemble
            return self.loss_and_results(scores, labels)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
