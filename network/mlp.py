from torch import nn
import torch
from utils import Activation
import torch.functional as F

# TODO: Remove preprocess function once I test with env.state2proxy() 

class MLP(nn.Module):
    def __init__(
        self,
        config,
        transformerCall=False,
        in_dim=512,
        hidden_layers=[512, 512],
        dropout_prob=0.0,
    ):

        super(MLP, self).__init__()
        self.config = config
        self.device = config.device
        act_func = "relu"

        self.input_max_length = self.config.env.max_len
        self.input_classes = self.config.env.dict_size
        self.out_dim = 1
        self.transformerCall = transformerCall

        if self.transformerCall == False:
            self.hidden_layers = [1024, 1024, 1024, 1024, 1024]
            self.dropout_prob = 0.0
        else:
            self.hidden_layers = hidden_layers
            self.dropout_prob = dropout_prob

        if self.transformerCall == False:
            self.init_layer_depth = int((self.input_classes) * (self.input_max_length))
        else:
            self.init_layer_depth = in_dim

        layers = [
            nn.Linear(self.init_layer_depth, self.hidden_layers[0]),
            Activation(act_func),
        ]
        layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.hidden_layers)):
            layers.extend(
                [
                    nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]),
                    Activation(act_func),
                    nn.Dropout(self.dropout_prob),
                ]
            )
        layers.append(nn.Linear(self.hidden_layers[-1], self.out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, input_len, mask):
        if self.transformerCall == False:
            x = self.preprocess(x)
        x = x.to(self.device)
        return self.model(x)

    def preprocess(self, inputs):
        inputs = inputs.to(torch.int64)
        inp_x = F.one_hot(inputs, num_classes=self.input_classes + 1)[:, :, :-1].to(
            torch.float32
        )
        inp = torch.zeros(inputs.shape[0], self.input_max_length, self.input_classes)
        inp[:, : inp_x.shape[1], :] = inp_x
        inputs = inp.reshape(inputs.shape[0], -1)
        return inputs
