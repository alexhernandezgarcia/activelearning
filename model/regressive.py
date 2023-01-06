import torch.nn as nn
from .mlp import ACTIVATION_KEY
import torch
import torch.nn.functional as F


class RegressiveMLP(nn.Module):
    def __init__(
        self,
        regressive_mode,
        base_num_hidden,
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
        self.dropout_fid = dropout_fid
        self.dropout_base = dropout_base
        self.base_num_hidden = base_num_hidden
        self.activation = ACTIVATION_KEY[activation]
        self.baseModule = nn.Sequential(
            nn.Linear(self.init_layer_depth, self.base_num_hidden),
            nn.Dropout(self.dropout_base),
            self.activation,
            nn.Linear(self.base_num_hidden, self.base_num_hidden),
            nn.Dropout(self.dropout_base),
            self.activation,
            nn.Linear(self.base_num_hidden, self.base_num_hidden),
            nn.Dropout(self.dropout_base),
            self.activation,
        )
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
                        nn.Dropout(self.dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(self.dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(self.dropout_fid),
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
                        nn.Dropout(self.dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(self.dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.fid_num_hidden),
                        nn.Dropout(self.dropout_fid),
                        self.activation,
                        nn.Linear(self.fid_num_hidden, self.num_output),
                    )
                    for i in range(self.num_fid)
                ]
            )

    def forward_seq_regressive(self, input, fid):
        """Implementation of DNN-MFBO"""
        input, fid_ohe = self.preprocess(input, fid)
        output_interm = self.baseModule(input)
        outputs = []
        for m in range(self.num_fid):
            augment_input = torch.cat([output_interm, input], axis=1)
            output_interm = self.layers[m](augment_input)
            outputs.append(output_interm)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid_ohe == 1]
        return output_masked

    def preprocess(self, input, fid):
        input = torch.zeros(input.shape[0], self.input_max_length * self.input_classes)
        input[:, : input.shape[1]] = input
        fid_ohe = F.one_hot(fid, num_classes=self.num_fid + 1)[:, 1:].to(torch.float32)
        fid_ohe = fid_ohe.to(self.device)
        return input, fid_ohe

    def forward_full_regressive(self, input, fid):
        """Implementation of BMBO-DARN"""
        input, fid_ohe = self.preprocess(input, fid)
        output_interm = self.baseModule(input)
        outputs = []
        augment_input = torch.cat([output_interm, inputs], axis=1)
        for m in range(self.num_fid):
            output_fid = self.layers[m](augment_input)
            augment_input = torch.cat([output_fid, augment_input], axis=1)
            outputs.append(output_fid)
        output_tensor = torch.stack(outputs, dim=1).to(self.device)
        output_masked = output_tensor[fid_ohe == 1]
        return output_masked
