import torch.nn as nn
from .mlp import ACTIVATION_KEY
import torch
import torch.nn.functional as F


class RegressiveMLP(nn.Module):
    def __init__(
        self,
        regressive_mode,
        base_num_hidden,
        base_num_layer,
        fid_num_hidden,
        activation,
        dropout_base,
        dropout_fid,
        num_output,
        n_fid,
        config_env,
        device,
    ):
        """
        Implements regressive deep learning networks in the mulit-fidelity setting as done Shibo Li et. al.
        For the emebedding variant, see nikita-0209/mf-proxy repo
        """
        super(RegressiveMLP, self).__init__()
        self.n_fid = n_fid
        input_classes = config_env.n_alphabet  # length
        input_max_length = config_env.max_seq_length  # n_dim
        self.init_layer_depth = int((input_classes) * (input_max_length))
        self.fid_num_hidden = fid_num_hidden
        self.num_output = num_output
        self.activation = ACTIVATION_KEY[activation]
        self.device = device
        self.base_hidden_layers = [base_num_hidden] * base_num_layer

        # base model
        base_layers = [
            nn.Linear(self.init_layer_depth, self.base_hidden_layers[0]),
            self.activation,
        ]
        base_layers += [nn.Dropout(dropout_base)]
        for i in range(1, len(self.base_hidden_layers)):
            base_layers.extend(
                [
                    nn.Linear(
                        self.base_hidden_layers[i - 1], self.base_hidden_layers[i]
                    ),
                    self.activation,
                    nn.Dropout(dropout_base),
                ]
            )
        base_layers.append(nn.Linear(self.base_hidden_layers[-1], self.num_output))
        self.base_module = nn.Sequential(*base_layers)

        # list of layers created
        if regressive_mode == "seq":
            self.forward = self.forward_seq_regressive
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.init_layer_depth + self.num_output,
                            self.fid_num_hidden,
                        ),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.num_output),
                    )
                    for i in range(self.n_fid)
                ]
            )
        elif regressive_mode == "full":
            self.forward = self.forward_full_regressive
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.init_layer_depth + (self.num_output + i),
                            self.fid_num_hidden,
                        ),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.num_output),
                    )
                    for i in range(self.n_fid)
                ]
            )

    # def preprocess(self, x, fid):
    #     """
    #     Args:
    #         x: one hot encoded tensor of shape (batch_size, obs_dim)
    #             paddded till the maximum el lwngth in the batch
    #         fid: tensor of shape (batch_size) with integers representing the fidelity
    #     Returns:
    #         input: tensor of shape (batch_size, init_layer_depth)
    #             padded till the maximum length in the dataset
    #         fid_ohe: one hot encoded represenation of fidelity (1 is represented as [1, 0] and 2 as [0, 1])
    #     """
    #     input = torch.zeros(x.shape[0], self.init_layer_depth)
    #     input[:, : x.shape[1]] = x
    #     fid_ohe = F.one_hot(fid, num_classes=self.num_fid + 1)[:, 1:].to(torch.float32)
    #     return input.to(x.device), fid_ohe

    def forward_seq_regressive(self, input):
        """Implementation of DNN-MFBO
        Args:
            input: one hot encoded input tensor
            fid: tensor of integers representing the fidelity
                example: tensor([2, 1])
        Returns:
            output_masked: tensor of shape (batch_size, num_output=1)"""
        # input, fid_ohe = self.preprocess(input, fid)
        x, fid = input[:, : -self.n_fid], input[:, -self.n_fid :]
        output_interm = self.base_module(x)
        outputs = []
        for m in range(self.n_fid):
            augment_input = torch.cat([output_interm, x], axis=1)
            output_interm = self.layers[m](augment_input)
            outputs.append(output_interm)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid == 1]
        return output_masked

    def forward_full_regressive(self, input):
        """Implementation of BMBO-DARN
        Args:
            input: one hot encoded input tensor
            fid: tensor of integers representing the fidelity
                example: tensor([2, 1])
        Returns:
            output_masked: tensor of shape (batch_size, num_output=1)"""
        x, fid_ohe = input[:, : -self.n_fid], input[:, -self.n_fid :]
        # input, fid_ohe = self.preprocess(input, fid)
        output_interm = self.base_module(x)
        outputs = []
        augment_input = torch.cat([output_interm, x], axis=1)
        for m in range(self.n_fid):
            output_fid = self.layers[m](augment_input)
            augment_input = torch.cat([output_fid, augment_input], axis=1)
            outputs.append(output_fid)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid_ohe == 1]
        return output_masked
