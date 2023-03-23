from gflownet.proxy.base import Proxy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers,
        dropout_prob,
        device,
        transformerCall=False,
        num_token=None,
        seq_len=None,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transformerCall = transformerCall
        self.d_numToken = num_token
        self.d_seqLength = seq_len
        self.device = device
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()]
        layers += [nn.Dropout(dropout_prob)]
        for i in range(1, len(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.transformerCall == False:
            x = self.preprocess(x)
        return self.model(x)

    def preprocess(self, inputs):
        inp_x = F.one_hot(inputs, num_classes=self.d_numToken + 1)[:, :, :-1].to(
            torch.float32
        )
        inp = torch.zeros(inputs.shape[0], self.d_seqLength, self.d_numToken)
        inp[:, : inp_x.shape[1], :] = inp_x
        inputs = inp.reshape(inputs.shape[0], -1).to(self.device).detach()
        return inputs


class AptamersMLP(Proxy):
    def __init__(self, cost, **kwargs):
        super().__init__(**kwargs)
        NUM_TOKEN = 4
        self.max_seq_length = 30
        self.cost = cost
        IN_DIM = self.max_seq_length * NUM_TOKEN
        NUM_OUTPUT = 1
        HIDDEN_FC = [1024, 1024, 1024, 1024, 1024]
        DROPOUT = 0.0
        # Came from the dataset creation in ActiveLearningPipeline
        # https://github.com/alexhernandezgarcia/ActiveLearningPipeline/blob/main/oracle.py#L376
        # as the indices were used to train the MLP
        # Amd the indices were assigned in the numbers2letters() function linked above
        self.lookup = {"A": 1, "T": 2, "C": 3, "G": 4}

        self.model = MLP(
            IN_DIM,
            NUM_OUTPUT,
            HIDDEN_FC,
            DROPOUT,
            device=self.device,
            transformerCall=False,
            num_token=NUM_TOKEN,
            seq_len=self.max_seq_length,
        )
        # self.model.load_state_dict(
        # "/home/mila/n/nikita.saxena/activelearning/storage/dna/length30/mlp_100k.pt"
        # )
        self.model.load_state_dict(
            torch.load(
                "/home/mila/n/nikita.saxena/activelearning/storage/dna/length30/mlp100k_correct.pt"
            )
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, sequences):
        """
        Args:
            x: torch.Tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor of shape (batch_size, output_dim)
        Transformation (one-hot etc) done within model
        """
        encoded_readable = [
            torch.tensor([self.lookup[el] for el in seq]) for seq in sequences
        ]
        states = torch.nn.utils.rnn.pad_sequence(
            encoded_readable, batch_first=True, padding_value=0.0
        )
        states = states.to(self.device)
        result = self.model(states)
        return result.squeeze(-1)
