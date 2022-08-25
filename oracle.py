import numpy as np

#for transition functions base2oracle, specific to each oracle
import torch
from torch import nn
import torch.nn.functional as F
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod

'''
Oracle Wrapper, callable in the AL pipeline, with key methods
'''
class Oracle:
    '''
    Generic Class for the oracle. 
    The different oracles (can be classes (MLP-toy oracle eg) or just a single function calling another annex program)
    can be called according to a config param in the method score
    '''
    def __init__(self, config):
        self.config = config
        #path to load and save the scored data
        self.path_data = config.path.data_oracle
        self.init_oracle()
    
    def init_oracle(self):
        #calling the relevant oracle. All oracles should follow the model of OracleBase
        if self.config.oracle.main == "mlp":
            self.oracle = OracleMLP(self.config)
        elif self.config.oracle.main == "toy":
            self.oracle = OracleToy(self.config)
        else:
            raise NotImplementedError

    def initialize_dataset(self, save = True, return_data = False):
        #the method to initialize samples in the BASE format is specific to each oracle for now. It can be changed.
        #the first samples are in the "base format", so as to be directly saved as such and sent for query (base2oracle transition) 

        samples = self.oracle.initialize_samples_base()

        data = {}
        data["samples"] = samples
        data["energies"] = self.score(samples)
        #print("data initialized", data)

        if save:
            np.save(self.path_data, data)
        if return_data:
            return data
       
    
    def score(self, queries):
        '''
        Calls the specific oracle (class/function) and apply its "get_score" method on the dataset
        '''
        scores = self.oracle.get_score(queries)
        return scores

    def update_dataset(self, queries, energies):
        dataset = np.load(self.path_data, allow_pickle = True)
        dataset = dataset.item()
        dataset["samples"] += queries
        dataset["energies"] += energies
        np.save(self.path_data, dataset)
        return



#For generalization purposes, the oracles are outside the main class of oracle
#First, general a general wrapper for oracles. All the other files.py follow the same formalism for modularity 
'''
BaseClass model for all other oracles
'''
class OracleBase:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def initialize_samples_base(self):
        pass

    @abstractmethod
    def base2oracle(self, state):
        '''
        Transition from base format (format of the queries and stored data) to a format callable by the oracle
        '''
        pass
    
    @abstractmethod
    def get_score(self, queries):
        '''
        - Transforms the queries to a good format with base2oracle
        - Calls the oracle on this data and return it in a good format --> for the proxy to train on.
        '''
        pass


'''
Different Oracles Implemented
'''

class OracleMLP(OracleBase):
    def __init__(self, config):
        super().__init__(config)
        #parameters useful for the format of this custom trained MLP
        self.dict_size = self.config.env.dict_size 
        self.max_len_mlp = 40 #this MLP was trained with seqs of len max 40
        self.path_oracle_mlp = self.config.path.model_oracle_MLP
        self.device = torch.device(self.config.device)
    
    def initialize_samples_base(self):
    
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len
        self.init_len = self.config.oracle.init_dataset.init_len
        self.random_seed = self.config.oracle.init_dataset.seed
        np.random.seed(self.random_seed)
        
        #random samples in base format
        samples = []
        for i in range(self.min_len, self.max_len + 1):
            #in order to have more long samples, we generate more random samples of longer sequences : it is arbitrary so far
            samples.extend(np.random.randint(0, self.dict_size, size = (int(self.init_len * i), i))) 
        
        np.random.shuffle(samples)
        samples = samples[:self.init_len]

        return samples

    def base2oracle(self, state):
        seq_array = state
        initial_len = len(seq_array)
        #transform the array as a tensor
        seq_tensor = torch.from_numpy(seq_array)
        #perform one hot encoding on it
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size + 1) #+1 for eos token
        seq_ohe = seq_ohe.reshape(1, -1).float()
        #adding the end of sequence token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()

        oracle_input = torch.cat((seq_ohe, eos_ohe), dim = 1)
        #adding 0-padding until max_len
        number_pads = self.max_len_mlp - initial_len
        if number_pads:
            padding = torch.cat([torch.tensor([0] * (self.dict_size + 1))] * number_pads).view(1, -1)
            oracle_input = torch.cat((oracle_input, padding), dim = 1)
        
        return oracle_input.to(self.device)[0]

    def get_score(self, queries):
        '''
        Input : list of arrays=seqs
        Outputs : list of floats=energies
        '''
        #list of tensors in ohe format
        list4oracle =  list(map(self.base2oracle, queries))
        #turn this into a single big tensor to call the MLP once
        tensor4oracle = torch.stack(list4oracle).view(len(queries), -1)

        #load the MLP oracle
        self.model = MLP()
        if os.path.exists(self.path_oracle_mlp):
            checkpoint = torch.load(self.path_oracle_mlp)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
        else:
            print("The MLP oracle could not be loaded")
            raise NotImplementedError
            
        self.model.eval()
        outputs = self.model(tensor4oracle)
    
        return outputs.squeeze(1).tolist()


class OracleToy(OracleBase):
    def __init__(self, config):
        super().__init__(config)
    
    def initialize_samples_base(self):
        self.dict_size = self.config.env.dict_size
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len

        self.init_len = self.config.oracle.init_dataset.init_len
        self.random_seed = self.config.oracle.init_dataset.seed
        np.random.seed(self.random_seed)
        
        #random samples in base format
        samples = []
        for i in range(self.min_len, self.max_len + 1):
            #in order to have more long samples, we generate more random samples of longer sequences : it is arbitrary so far
            samples.extend(np.random.randint(0, self.dict_size, size = (int(self.init_len * i), i))) 
        
        np.random.shuffle(samples)
        samples = samples[:self.init_len]

        return samples

    def base2oracle(self, state):
        '''
        Input : array = seq
        Output : array = seq
        '''
        seq = state
        return seq

    def get_score(self, queries):
        count_zero = lambda x : float(np.count_nonzero(x == 0))
        outputs = list(map(count_zero, queries))
        return outputs



### Diverse Models of Oracle for now
'''
ORACLE MODELS ZOO
'''
###

class Activation(nn.Module):
    def __init__(self, activation_func = "relu"):
        super().__init__()
        if activation_func == "relu":
            self.activation = F.relu
        elif activation_func == "gelu":
            self.activation = F.gelu

    def forward(self, input):
        return self.activation(input)


#This MLP was specifically trained on a special dataset
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = 2
        self.filters = 128
        self.init_layer_depth = int(
            (40 + 1) * (4 + 1)
        )  

        # build input and output layers
        self.initial_layer = nn.Linear(
            int(self.init_layer_depth), self.filters
        )
        self.activation1 = Activation("gelu")
        self.output_layer = nn.Linear(self.filters, 1, bias=False)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.dropouts = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters, self.filters))
            self.activations.append(Activation("gelu"))
            self.dropouts.append(nn.Dropout(p=0.1))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, x):

        x = self.activation1(
            self.initial_layer(x)
        )  # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)

        y = self.output_layer(x)

        return y

