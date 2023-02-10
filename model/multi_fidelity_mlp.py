import torch.nn as nn
from .mlp import ACTIVATION_KEY
import torch
import torch.nn.functional as F


class MultiFidelityMLP(nn.Module):
    def __init__(
        self,
        shared_head,
        base_num_hidden,
        base_num_layer,
        fid_num_hidden,
        fid_num_layer,
        activation,
        dropout_prob,
        num_output,
        n_fid,
        config_env,
        device,
    ):
        super(MultiFidelityMLP, self).__init__()

        input_max_length = config_env.max_seq_length  # config_env.n_dim
        input_classes = config_env.n_alphabet  # config_env.length
        self.num_output = num_output
        self.init_layer_depth = int((input_classes) * (input_max_length))

        self.embed_hidden_layers = [base_num_hidden] * base_num_layer
        self.fid_hidden_layers = [fid_num_hidden] * fid_num_layer
        self.activation = ACTIVATION_KEY[activation]
        self.n_fid = n_fid
        self.device = device

        embed_layers = [
            nn.Linear(self.init_layer_depth, self.embed_hidden_layers[0]),
            self.activation,
        ]
        embed_layers += [nn.Dropout(dropout_prob)]
        for i in range(1, len(self.embed_hidden_layers)):
            embed_layers.extend(
                [
                    nn.Linear(
                        self.embed_hidden_layers[i - 1], self.embed_hidden_layers[i]
                    ),
                    self.activation,
                    nn.Dropout(dropout_prob),
                ]
            )
        self.embed_module = nn.Sequential(*embed_layers)

        fid_layers = [
            nn.Linear(
                self.embed_hidden_layers[-1] + self.n_fid * (shared_head),
                self.fid_hidden_layers[0],
            ),
            self.activation,
        ]
        fid_layers += [nn.Dropout(dropout_prob)]
        for i in range(1, len(self.fid_hidden_layers)):
            fid_layers.extend(
                [
                    nn.Linear(self.fid_hidden_layers[i - 1], self.fid_hidden_layers[i]),
                    self.activation,
                    nn.Dropout(dropout_prob),
                ]
            )
        fid_layers.append(nn.Linear(self.fid_hidden_layers[-1], self.num_output))

        if shared_head == False:
            self.forward = self.forward_multiple_head
            for i in range(self.n_fid):
                setattr(self, f"fid_{i}_module", nn.Sequential(*fid_layers))
        else:
            self.fid_module = nn.Sequential(*fid_layers)
            self.forward = self.forward_shared_head

    # def preprocess(self, x, fid):
    #     # input = torch.zeros(x.shape[0], self.init_layer_depth)
    #     # input[:, : x.shape[1]] = x
    #     fid_ohe = F.one_hot(fid, num_classes=self.n_fid + 1)[:, 1:].to(torch.float32)
    #     return input.to(x.device), fid_ohe

    def forward_shared_head(self, input):
        x, fid = input[:, : self.n_fid], input[:, -self.n_fid :]
        state = torch.zeros(x.shape[0], self.init_layer_depth).to(self.device)
        state[:, : x.shape[1]] = x
        # self.preprocess(input, fid)
        embed = self.embed_module(state)
        embed_with_fid = torch.concat([embed, fid], dim=1)
        out = self.fid_module(embed_with_fid)
        return out

    def forward_multiple_head(self, input):
        """
        Currently compatible with at max two fidelities.
        Need to extend functionality to more fidelities if need be.
        """
        x, fid_ohe = input[:, : self.n_fid], input[:, self.n_fid :]
        state = torch.zeros(x.shape[0], self.init_layer_depth).to(self.device)
        state[:, : x.shape[1]] = x
        # x, fid_ohe = self.preprocess(inputs, fid)
        embed = self.embed_module(state)
        output_fid_1 = self.fid_0_module(embed)
        output_fid_2 = self.fid_1_module(embed)
        out = torch.stack([output_fid_1, output_fid_2], dim=-1).squeeze(-2)
        output = out[fid_ohe == 1].unsqueeze(dim=-1)
        return output
