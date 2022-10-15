from heapq import heapify
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import os
from abc import abstractmethod
from torch.autograd import Variable 
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

'''
GLOBAL CLASS PROXY WRAPPER, callable with key methods
'''
class Proxy:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.init_proxy()

    def init_proxy(self):
        #for now only the MLP proxy is implemented, but another class inheriting from ProxyBase has to be created for another proxy (transformer, ...)
        if self.config.proxy.model.lower() == "mlp":
            self.proxy = ProxyMLP(self.config, self.logger)
        elif self.config.proxy.model.lower() == "lstm":
            self.proxy = ProxyLSTM(self.config, self.logger)
        elif self.config.proxy.model.lower() == "transformer":
            self.proxy = ProxyTransformer(self.config, self.logger)
        else:
            raise NotImplementedError
    
    def train(self):
        self.data_handler = BuildDataset(self.config, self.proxy)
        self.proxy.converge(self.data_handler)
        return


'''
PROXY OBJECTS
'''    
class ProxyBase:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.device = self.config.device
        self.path_data = self.config.path.data_oracle
        self.path_model = self.config.path.model_proxy  

        #Training Parameters
        self.training_eps = self.config.proxy.training.eps
        self.max_epochs = self.config.proxy.training.max_epochs
        self.history = self.config.proxy.training.history
        assert self.history <= self.max_epochs
        self.dropout = self.config.proxy.training.dropout
        self.batch_size = self.config.proxy.training.training_batch

        #Dataset management
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed

        self.model_class = NotImplemented  #will be precised in child classes

 
    @abstractmethod
    def init_model(self):
        '''
        Initialize the proxy we want (cf config). Each possible proxy is a class (MLP, transformer ...)
        Ensemble methods will be another separate class
        '''
        self.model = self.model_class(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad = True)

    @abstractmethod
    def load_model(self, dir_name = None):
        '''
        will not have to be used normally because the global object ActiveLearning.proxy will be continuously updated
        '''
        if dir_name == None:
            dir_name = self.config.path.model_proxy
        
        self.init_model()
        
        if os.path.exists(dir_name):
            checkpoint = torch.load(dir_name)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.device == "cuda":
                self.model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        
        else:
            raise NotImplementedError
    
    @abstractmethod
    def converge(self, data_handler):
        '''
        will call getDataLoaders/train_batch/ test / checkConvergencce 
        '''
        #we reset the model, cf primacy bias, here we train on more and more data
        self.init_model()

        #for statistics we save the tr and te errors
        [self.err_tr_hist, self.err_te_hist] = [[], []]
        
        #get training data in torch format
        tr, te = data_handler.get_data_loaders()

        self.converged = 0
        self.epochs = 0

        while self.converged != 1:

            if (
                self.epochs > 0
            ):  #  this allows us to keep the previous model if it is better than any produced on this run
                self.train(tr)  # already appends to self.err_tr_hist
            else:
                self.err_tr_hist.append(0)

            self.test(te)


            if self.err_te_hist[-1] == np.min(
                self.err_te_hist
            ):  # if this is the best test loss we've seen
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.path_model,
                )  # we update only the best, not keep the previous ones
                print(
                    "new best found at epoch {}, of value {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )

            # after training at least "history" epochs, check convergence
            if self.epochs >= self.history + 1:
                self.check_convergence()

            if (self.epochs % 10 == 0) and self.config.debug:
                print(
                    "Model epoch {} test loss {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )  

            self.epochs += 1

            #TODO : implement comet logger (with logger object in activelearning.py)
            # if self.converged == 1:
            #     self.statistics.log_comet_proxy_training(
            #         self.err_tr_hist, self.err_te_hist
            #     )


    @abstractmethod
    def train(self, tr):
        ''' 
        will call getLoss
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.get_loss(trainData)
            err_tr.append(loss.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.err_te_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())
    
    @abstractmethod
    def test(self, te):
        err_te = []
        self.model.eval()
        with torch.no_grad():
            for i, testData in enumerate(te):
                loss = self.get_loss(testData)
                err_te.append(loss.data)
        
        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())
    
    @abstractmethod
    def get_loss(self, data):
        inputs = data[0]
        targets = data[1]
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        # output = self.model(inputs.float())
        output = self.model(inputs)
        loss = F.mse_loss(output[:, 0], targets.float())
        return loss
        
    @abstractmethod
    def check_convergence(self):
        eps = self.training_eps
        history = self.history
        max_epochs = self.max_epochs

        if all(
            np.asarray(self.err_te_hist[-history + 1 :]) > self.err_te_hist[-history]
        ):  # early stopping
            self.converged = 1  # not a legitimate criteria to stop convergence ...
            print(
                "Model converged after {} epochs - test loss increasing at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )  
            )

        if (
            abs(self.err_te_hist[-history] - np.average(self.err_te_hist[-history:]))
            / self.err_te_hist[-history]
            < eps
        ):
            self.converged = 1
            print( 
                "Model converged after {} epochs - hit test loss convergence criterion at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )

        if self.epochs >= max_epochs:
            self.converged = 1
            print(
                "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )
      
    @abstractmethod
    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data).cpu().detach().numpy()
            return output  

    @abstractmethod
    def base2proxy(self, state):
        return state
        # pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

'''
In the child Classes, the previous abstract methods can be overwritten. In what follows, the minimum is done to precise the proxy, ie
- the Network is precised
- The conversion format is given for its input
'''
class ProxyMLP(ProxyBase):
    def __init__(self, config, logger, init_model = False):
        super().__init__(config, logger)
        self.model_class = MLP
        self.device = config.device
        if init_model:
            self.init_model()

    def init_model(self):
        super().init_model()
            
    def load_model(self, dir_name = None):
        super().load_model(dir_name)
   
    def converge(self, data_handler):
        super().converge(data_handler)

    def train(self, tr):
        super().train(tr)

    def test(self, te):
        super().test(te)

    def get_loss(self, data):
        return super().get_loss(data)
    
    def check_convergence(self):
        super().check_convergence()

    def evaluate(self, data):
        return super().evaluate(data)


class ProxyLSTM(ProxyBase):
    def __init__(self, config, logger, init_model = False):
        super().__init__(config, logger)
        self.model_class = LSTM
        self.device = config.device
        if init_model:
            self.init_model()

    def init_model(self):
        super().init_model()
            
    def load_model(self, dir_name = None):
        super().load_model(dir_name)
   
    def converge(self, data_handler):
        super().converge(data_handler)

    def train(self, tr):
        super().train(tr)

    def test(self, te):
        super().test(te)

    def get_loss(self, data):
        inputs = data[0]
        targets = data[1]
        inputLens = data[2]
        inputs = inputs.to(self.device)
        inputLens = inputLens.to(self.device)
        targets = targets.to(self.device)
        output = self.model(inputs, inputLens)
        loss = F.mse_loss(output[:, 0], targets.float())
        return loss
    
    def check_convergence(self):
        super().check_convergence()

    def evaluate(self, data):
        return super().evaluate(data)

        

class ProxyTransformer(ProxyBase):
    def __init__(self, config, logger, init_model = False):
        super().__init__(config, logger)
        self.model_class = Transformer
        if init_model:
            self.init_model()

    def init_model(self):
        super().init_model()
            
    def load_model(self, dir_name = None):
        super().load_model(dir_name)
   
    def converge(self, data_handler):
        super().converge(data_handler)

    def train(self, tr):
        super().train(tr)

    def test(self, te):
        super().test(te)

    def get_loss(self, data):
        inputs = data[0]
        targets = data[1]
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        output = self.model(inputs, None)
        return F.mse_loss(output[:, 0], targets.float())
    
    def check_convergence(self):
        super().check_convergence()

    def evaluate(self, data):
        return super().evaluate(data)
  
    def base2proxy(self, state):
        return super().base2proxy(state)


'''
BuildDataset utils
'''
class BuildDataset:
    '''
    Will load the dataset scored by the oracle and convert it in the right format for the proxy with transition.base2proxy
    '''
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.path_data = self.config.path.data_oracle
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed

        self.load_dataset()
    
    
    def load_dataset(self):
        dataset = np.load(self.path_data, allow_pickle = True)
        dataset = dataset.item()
        
        #Targets of training
        self.targets = np.array(dataset["energies"])
        self.targets = (self.targets - np.mean(self.targets))/np.std(self.targets)
        #Samples of training
        samples = dataset['samples']
        self.samples = samples 
    
    def reshuffle(self):
        self.samples, self.targets = shuffle(
            self.samples, self.targets, random_state=self.seed_data
        )
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


    def collate_batch(self, batch):
        y, x, = [], []
        for (_text,_label) in batch:
            y.append(_label)
            x.append(torch.tensor(_text))
        y = torch.tensor(y, dtype=torch.float)
        xPadded = pad_sequence(x, batch_first=True, padding_value=0.0)
        lens = torch.LongTensor([len(i) for i in x])
        return xPadded, y, lens


    def get_data_loaders(self):

        if self.shuffle_data:
            self.reshuffle()
        
        train_size = int(0.8 * self.__len__())
        test_size = self.__len__() - train_size

        train_dataset = []
        test_dataset = []

        for i in range(
            test_size, test_size + train_size
        ):  # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
            train_dataset.append(self.__getitem__(i))
        for i in range(test_size):  # test data is drawn from oldest datapoints
            test_dataset.append(self.__getitem__(i))

        tr = data.DataLoader(
            train_dataset,
            batch_size = self.proxy.batch_size,
            shuffle = True,
            num_workers= 0,
            pin_memory= False, 
            collate_fn=self.collate_batch
        )

        te = data.DataLoader(
            test_dataset,
            batch_size = self.proxy.batch_size,
            shuffle = False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch
        )
    
        return tr, te
    


###
'''
MODEL ZOO
'''
###

class Activation(nn.Module):
    def __init__(self, activation_func):
        super().__init__()
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "gelu":
            self.activation = F.gelu

    def forward(self, input):
        return self.activation(input)

class MLP(nn.Module):
    def __init__(self, config, transformerCall=False, in_dim=512, hidden_layers=[512, 512], dropout_prob=0.0):

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

        layers = [nn.Linear(self.init_layer_depth, self.hidden_layers[0]), Activation(act_func)] 
        layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.hidden_layers)):
            layers.extend([nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]), Activation(act_func), nn.Dropout(self.dropout_prob)])
        layers.append(nn.Linear(self.hidden_layers[-1], self.out_dim))
        self.model = nn.Sequential(*layers)
     
    def forward(self, x):
        if self.transformerCall == False:
            x = self.preprocess(x)
        return self.model(x)

    def preprocess(self, inputs):
        inp_x = F.one_hot(inputs, num_classes=self.input_classes+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(inputs.shape[0], self.input_max_length, self.input_classes)
        inp[:, :inp_x.shape[1], :] = inp_x
        inputs = inp.reshape(inputs.shape[0], -1)
        return inputs


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        act_func = "relu"
        self.config = config
        self.device = config.device
        self.input_classes = self.config.env.dict_size  #number of classe
        self.hidden_size_fc = 1024 #input size
        self.hidden_size_lstm = 256 #input size
        self.num_layer = 2 #hidden state
        self.max_seq_length = self.config.env.max_len #sequence length
        self.num_output = 1
        self.dropout_prob = 0.0
        self.bidirectional = False

        self.lstm = nn.LSTM(input_size=self.input_classes, hidden_size=self.hidden_size_lstm,
                          num_layers=self.num_layer, batch_first=True, dropout = self.dropout_prob, bidirectional = self.bidirectional) #lstm
        self.fc_1 =  nn.LazyLinear(self.hidden_size_fc)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size_fc, self.num_output) #fully connected last layer

        self.activation = Activation(act_func)
    
    def forward(self ,inputs, inputLens):
        x = self.preprocess(inputs)
        xPack = pack_padded_sequence(x, inputLens, batch_first=True, enforce_sorted=False)
        h_0 = Variable(torch.zeros(self.num_layer, inputs.size(0), self.hidden_size_lstm)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layer, inputs.size(0), self.hidden_size_lstm)).to(self.device) #internal state
        outputPack, (hn, cn) = self.lstm(xPack, (h_0, c_0)) 
        output, outputLens = pad_packed_sequence(outputPack, batch_first=True, total_length=self.max_seq_length) 
        output = output.view(inputs.size(0), -1)
        out = self.fc_1(output) #first Dense
        out = self.dropout(out) 
        out = self.activation(out) #relu
        out = self.fc(out)
        return out

    def preprocess(self, inputs):
        inp_x = F.one_hot(inputs, num_classes=self.input_classes+1)[:, :, 1:].to(torch.float32)
        inp = torch.zeros(inputs.shape[0], self.max_seq_length, self.input_classes)
        inp[:, :inp_x.shape[1], :] = inp_x
        inputs = inp
        return inputs

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.device = config.device
        self.input_classes = self.config.env.dict_size  #number of classe
        self.max_seq_length = self.config.env.max_len #sequence length
        self.num_hidden = 256
        self.dropout_prob = 0.0
        self.num_head = 8
        self.pre_ln = True
        self.num_layers = 4
        self.factor = 2
        self.num_outputs = 1
        self.pos = PositionalEncoding(self.num_hidden, dropout=self.dropout_prob, max_len=self.max_seq_length + 2)
        self.embedding = nn.Embedding(self.input_classes+1, self.num_hidden)
        
        encoder_layers = nn.TransformerEncoderLayer(self.num_hidden, self.num_head, self.num_hidden, dropout = self.dropout_prob, norm_first = self.pre_ln)
        self.encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        self.output = MLP(config, True, self.num_hidden, [self.factor * self.num_hidden, self.factor * self.num_hidden], self.dropout_prob)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = self.num_hidden

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def forward(self, x, mask):
        x = self.preprocess(x)
        x = self.embedding(x) 
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, torch.arange(x.shape[1])]
        y = self.output(pooled_x)
        return y

    def preprocess(self, inputs):
        inputs = torch.transpose(inputs, 1, 0)
        return inputs