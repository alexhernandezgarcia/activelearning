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
        num_fid,
        config_env,
    ):
        """
        Implements regressive deep learning networks in the mulit-fidelity setting as done Shibo Li et. al. 
        For the emebedding variant, see nikita-0209/mf-proxy repo
        """
        super(RegressiveMLP, self).__init__()
        self.num_fid = num_fid
        input_classes = config_env.length  
        input_max_length = config_env.n_dim  
        self.init_layer_depth = int((input_classes) * (input_max_length))
        self.fid_num_hidden = fid_num_hidden
        self.num_output = num_output
        self.activation = ACTIVATION_KEY[activation]

        self.base_hidden_layers = [base_num_hidden] * base_num_layer

        # base model
        base_layers = [
            nn.Linear(self.init_layer_depth, self.base_hidden_layers[0]),
            self.activation,
        ]
        base_layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.base_hidden_layers)):
            base_layers.extend(
                [
                    nn.Linear(self.base_hidden_layers[i - 1], self.base_hidden_layers[i]),
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
                            self.input_dim + (self.num_output * i),
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
                    for i in range(self.num_fid)
                ]
            )
        elif regressive_mode == "full":
            self.forward = self.forward_full_regressive
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.input_dim + (self.num_output + i) * i,
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
                    for i in range(self.num_fid)
                ]
            )

    def preprocess(self, input, fid):
        input = torch.zeros(input.shape[0], self.input_max_length * self.input_classes)
        input[:, : input.shape[1]] = input
        fid_ohe = F.one_hot(fid, num_classes=self.num_fid + 1)[:, 1:].to(torch.float32)
        fid_ohe = fid_ohe.to(self.device)
        return input, fid_ohe
    
    def forward_seq_regressive(self, input, fid):
        """Implementation of DNN-MFBO"""
        input, fid_ohe = self.preprocess(input, fid)
        output_interm = self.base_module(input)
        outputs = []
        for m in range(self.num_fid):
            augment_input = torch.cat([output_interm, input], axis=1)
            output_interm = self.layers[m](augment_input)
            outputs.append(output_interm)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid_ohe == 1]
        return output_masked

    def forward_full_regressive(self, input, fid):
        """Implementation of BMBO-DARN"""
        input, fid_ohe = self.preprocess(input, fid)
        output_interm = self.base_module(input)
        outputs = []
        augment_input = torch.cat([output_interm, inputs], axis=1)
        for m in range(self.num_fid):
            output_fid = self.layers[m](augment_input)
            augment_input = torch.cat([output_fid, augment_input], axis=1)
            outputs.append(output_fid)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid_ohe == 1]
        return output_masked
