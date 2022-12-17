import torch.nn as nn
from base import ACTIVATION_KEY
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, config, env):
        super(LSTM, self).__init__()

        self.config = config
        # Env-specific parameters
        self.input_classes = env.dict_size  # number of classes
        self.max_seq_length = env.max_len  # sequence length
        # Network
        self.hidden_dim_fc = self.config.hidden_dim_fc  # input size
        self.hidden_dim_lstm = self.config.hidden_dim_lstm  # input size
        self.num_layer = self.config.num_layer  # hidden state
        self.num_output = self.config.num_output
        self.dropout_prob = self.config.dropout_prob        

        self.lstm = nn.LSTM(
            input_size=self.input_classes,
            hidden_size=self.hidden_dim_lstm,
            num_layers=self.num_layer,
            batch_first=True,
            dropout=self.dropout_prob,
            bidirectional=self.bidirectional,
        ) 
        self.fc_1 = nn.LazyLinear(self.hidden_dim_fc)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(
            self.hidden_dim_fc, self.num_output
        )

        self.activation = ACTIVATION_KEY[config.activation]

    def forward(self, x):
        """
        Args:
            tensor of dimension batch x max_seq_length_in_batch x k, 
            where k is num_tokens if input is required to be one-hot encoded
        """
        x_len = list(map(len, x))
        # Pads the tensor till maximum length
        input = torch.zeros(x.shape[0], self.input_max_length, self.input_classes)
        input[:, : x.shape[1], :] = x
        # Reshapes the tensor to batch_size x init_layer_depth
        x = input.reshape(x.shape[0], -1)
        # Performs a forward call

        xPack = pack_padded_sequence(
            x, x_len, batch_first=True, enforce_sorted=False
        )
        h_0 = Variable(
            torch.zeros(self.num_layer, x.size(0), self.hidden_dim_lstm)
        ).to(
            self.device
        )  # hidden state
        c_0 = Variable(
            torch.zeros(self.num_layer, x.size(0), self.hidden_dim_lstm)
        ).to(
            self.device
        )  # internal state
        outputPack, (hn, cn) = self.lstm(xPack, (h_0, c_0))
        output, outputLens = pad_packed_sequence(
            outputPack, batch_first=True, total_length=self.max_seq_length
        )
        output = output.view(x.size(0), -1)
        out = self.fc_1(output)  # first Dense
        out = self.dropout(out)
        out = self.activation(out)  # relu
        out = self.fc(out)
        return out