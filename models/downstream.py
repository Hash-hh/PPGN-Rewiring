import torch
import torch.nn as nn
import layers.layers as layers
import layers.modules as modules


class Downstream(nn.Module):
    def __init__(self, config):
        """
        Build the model computation graph, until scores/values are returned at the end
        """
        super().__init__()

        self.config = config
        use_new_suffix = config.architecture.new_suffix  # True or False
        block_features = config.architecture.block_features  # List of number of features in each regular block
        original_features_num = (config.node_labels + 1 + 1) * 2  # Number of features of the input # last +1 is for edge weight, *2 is for cat of a and b

        # First part - sequential mlp blocks
        last_layer_features = original_features_num
        self.reg_blocks = nn.ModuleList()
        for layer, next_layer_features in enumerate(block_features):
            mlp_block = modules.RegularBlock(config, last_layer_features, next_layer_features)
            self.reg_blocks.append(mlp_block)
            last_layer_features = next_layer_features

        # Second part
        self.fc_layers = nn.ModuleList()
        if use_new_suffix:
            for output_features in block_features:
                # each block's output will be pooled (thus have 2*output_features), and pass through a fully connected
                fc = modules.FullyConnected(2*output_features, self.config.num_classes, activation_fn=None)
                self.fc_layers.append(fc)

        else:  # use old suffix
            # Sequential fc layers
            self.fc_layers.append(modules.FullyConnected(2*block_features[-1], 512))
            self.fc_layers.append(modules.FullyConnected(512, 256))
            self.fc_layers.append(modules.FullyConnected(256, self.config.num_classes, activation_fn=None))

    def forward(self, repeated_data, new_data):
        a = repeated_data
        a_padding = torch.zeros(a.size(0), 1, a.size(2), a.size(3), device=a.device)  # padding for edge weight (no edge weights in the original data)
        a = torch.cat([a, a_padding], dim=1)

        b = new_data
        x = torch.cat([a, b], dim=1)
        scores = torch.tensor(0, device=repeated_data.device, dtype=a.dtype)

        for i, block in enumerate(self.reg_blocks):

            x = block(x)

            if self.config.architecture.new_suffix:
                # use new suffix
                scores = self.fc_layers[i](layers.diag_offdiag_maxpool(x)) + scores

        if not self.config.architecture.new_suffix:
            # old suffix
            x = layers.diag_offdiag_maxpool(x)  # NxFxMxM -> Nx2F
            for fc in self.fc_layers:
                x = fc(x)
            scores = x

        return scores
