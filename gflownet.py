import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import pandas as pd
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod

#Utils function for the whole file, fixed once and for all
global tf_tensor, tf_list, tl_tensor, _dev
_dev = [torch.device("cpu")]
tf_list = lambda x: torch.FloatTensor(x).to(_dev[0])
tl_list = lambda x: torch.LongTensor(x).to(_dev[0])
to = lambda x : x.to(_dev[0])

def set_device(dev):
    _dev[0] = dev

'''
GFlownet Wrapper
'''
class GFlowNet:
    def __init__(self, config, logger, env):
        self.config = config
        self.logger = logger
        self.env = env

        self.device = self.config.device
        set_device(self.device)

        self.init_gflownet()
    
    def init_gflownet(self):
        #for now there is only one GFlowNet - configuration called A, but there will be several with multifidelity (several possibilities of sampling m)
        self.gflownet = GFlowNet_A(self.config, self.logger, self.env)
        return
    
    def train(self):
        return
        


'''
GFlowNet Objects
'''
class GFlowNetBase:
    def __init__(self, config, logger, env, load_best_model = False):
        self.config = config
        self.logger = logger
        self.env = env.env
        
        #set loss function, device, buffer ...
        self.path_model = self.config.path.model_gfn
        self.device = self.config.device

        if self.config.gflownet.loss.function == "flowmatch":
            self.loss_function = self.flowmatch_loss
        elif self.config.gflownet.loss.function == "trajectory_balance":
            self.loss_function = self.trajectory_balance
        else:
            raise NotImplementedError  
        
        self.get_model_class()
        if load_best_model:
            self.load_best_model()
        


    @abstractmethod
    def get_model_class(self):
        if self.config.gflownet.policy_model == "mlp":
            self.model_class = MLP
        else:
            raise NotImplementedError


    @abstractmethod
    def make_model(self, new_model = False, best_model = False):
        '''
        Initializes the GFN policy network (separate class for MLP for now), and load the best one (random if not best GFN yet)
        '''
        pass

    @abstractmethod
    def load_best_model(self):
        pass   
    
    @abstractmethod
    def get_training_data(self):
        '''
        Calls the buffer to get some interesting training data
        Performs backward sampling for off policy data and forward sampling
        Calls the utils method forward sampling and backward sampling
        '''
        pass
    
    def forward_sampling(self):
        return
    
    def backward_sampling(self):
        return
    
    def flowmatch_loss(self):
        return
    
    def trajectory_balance(self):
        return
    
    def train(self):
        return
    
    def sample(self):
        '''
        Just performs forward sampling with the trained GFlownet
        '''
        return

###-----------SUBCLASS OF SPECIFIC GFLOWNET-----------------------
class GFlowNet_A(GFlowNetBase):
    def __init__(self, config, logger, env):
        super().__init__(config, logger, env)

        self.buffer = Buffer(self.config)

    def make_model(self, new_model=False, best_model=False):
        super().make_model(new_model, best_model)
        def make_opt(params, config):
            params = list(params)
            if not len(params):
                return None
            if config.gflownet.training_process.opt == "adam":
                opt = torch.optim.Adam(
                    params,
                    config.gflownet.training_process.learning_rate,
                    betas=(
                        config.gflownet.training_process.adam_beta1,
                        config.gflownet.training_process.adam_beta2,
                    ),
                )

            elif config.gflownet.training.opt == "msgd":
                opt = torch.optim.SGD(
                    params,
                    config.gflownet.training_process.learning_rate,
                    momentum=config.gflownet.training_process.momentum,
                )

            else:
                raise NotImplementedError
            return opt

        def make_lr_scheduler(optimizer, config):
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.gflownet.training_process.lr_decay_period,
                gamma=config.gflownet.training_process.lr_decay_gamma,
            )
            return lr_scheduler
        
        if new_model:
            self.model = self.model_class(self.config)
            self.opt = make_opt(self.model.parameters(), self.config)

            if self.device == "cuda":
                self.model.cuda()  # move net to GPU
                for state in self.opt.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            
            self.lr_scheduler = make_lr_scheduler(self.opt, self.config)
        
        if best_model:
            path_best_model = self.path_model
            if os.path.exists(path_best_model):
                checkpoint = torch.load(path_best_model)
                self.best_model = self.model_class(self.config)
                self.best_model.load_state_dict(checkpoint["model_state_dict"])
                self.best_opt = make_opt(self.best_model.parameters(), self.config)
                self.best_opt.load_state_dict(checkpoint["optimizer_state_dict"])
                print("best gfn loaded") 

                if self.device == "cuda":
                    self.best_model.cuda()  # move net to GPU
                    for state in self.best_opt.state.values():  # move optimizer to GPU
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()   

                self.best_lr_scheduler = make_lr_scheduler(self.best_opt, self.config)    
            else:
                print("the best previous model could not be loaded")
                self.best_model = self.model_class(self.config)
                self.best_opt = make_opt(self.best_model.parameters(), self.config)
                self.best_lr_scheduler = make_lr_scheduler(self.best_opt, self.config)
    
    def forward_sampling(self):
        super().forward_sampling()
    def forward_sample(self, envs, times, policy="model", model=None, temperature=1.0):
        """
        Performs a forward action on each environment of a list.

        Args
        ----
        env : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        policy : string
            - model: uses self.model to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if policy="model"

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        if not isinstance(envs, list):
            envs = [envs]
        states = [env.state2obs() for env in envs]
        mask_invalid_actions = [env.get_mask_invalid_actions() for env in envs]
        random_action = self.rng.uniform()
        t0_a_model = time.time()
        if policy == "model":
            with torch.no_grad():
                action_logits = model(tf(states))
            action_logits /= temperature
        elif policy == "uniform":
            action_logits = tf(np.zeros(len(states)), len(self.env.action_space) + 1)
        else:
            raise NotImplemented
        if self.mask_invalid_actions:
            action_logits[torch.tensor(mask_invalid_actions)] = -1000
        if all(torch.isfinite(action_logits).flatten()):
            actions = Categorical(logits=action_logits).sample()
        else:
            if self.debug:
                raise ValueError("Action could not be sampled from model!")
        t1_a_model = time.time()
        times["actions_model"] += t1_a_model - t0_a_model
        assert len(envs) == actions.shape[0]
        # Execute actions
        _, _, valids = zip(*[env.step(action) for env, action in zip(envs, actions)])
        return envs, actions, valids
    

    def backward_sampling(self):
        return super().backward_sampling()
    
    def get_training_data(self, batch_size):
        super().get_training_data()

        batch = []
        envs = [
            self.env.env_class(self.config, self.env.acquisition, idx = idx)
            for idx in range(batch_size)
        ]

        #OFFLINE DATA
        self.buffer.make_train_test_set()
        self.rng = np.random.default_rng(self.config.gflownet.training_data.seed)
        offline_samples = int(self.config.gflownet.training_data.pct_batch_empirical * len(envs))
        for env in envs[:offline_samples]:
            state = self.rng.permutation(self.buffer.train.samples.values)[0]
            state_manip = self.env.base2manip(state, done = True)

            env.done = True
            env.seq = state

            action = self.env.token_eos

            while len(env.seq) > 0:
                previous_done = env.done
                previous_mask = env.get_mask()

                parents, parents_a = env.get_parents(backward = True)
                            





        
        




'''
Utils Buffer
'''

class Buffer:
    '''
    BUffer of data : 
    - loads the data from oracle and put the best ones as offline training data
    - maintains a replay buffer composed of the best trajectories sampled for training
    '''
    def __init__(self, config):
        self.config = config
        self.path_data_oracle = self.config.path.data_oracle
    

    def np2df(self):
        data_dict = np.load(self.path_data_oracle, allow_pickle = True).item()
        seqs = data_dict["samples"]
        energies = data_dict["energies"]
        df = pd.DataFrame(
            {
                "samples" : seqs,
                "energies" : energies,
                "train" : [False] * len(seqs)
                "test" :  [False] * len(seqs)
            }
        )

        return df
    
    def make_train_test_set(self):
        df = self.np2df()
        rng = np.random.default_rng(47)
        indices = rng.permutation(len(df.index))
        n_tt = int(0.1 * len(indices))
        indices_tt = indices[:n_tt]
        indices_tr = indices[n_tt:]
        df.loc[indices_tt, "test"] = True
        df.loc[indices_tr, "train"] = True

        self.train = df.loc[df.train]
        self.test = df.loc[df.test]





'''
Model Zoo
'''



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
    def __init__(self, config):
        super().__init__()
     
        self.config = config
        act_func = "relu"

        # Architecture
        self.input_max_length = self.config.env.max_len + 1  
        self.input_classes = self.config.env.dict_size + 1  
        self.init_layer_size = int(self.input_max_length * self.input_classes)
        self.final_layer_size = int(self.input_classes)  

        self.filters = 256
        self.layers = 16

        prob_dropout = self.config.gflownet.training_process.dropout

        # build input and output layers
        self.initial_layer = nn.Linear(
            self.init_layer_size, self.filters
        )  
        self.initial_activation = Activation(act_func)
        self.output_layer = nn.Linear(
            self.filters, self.final_layer_size, bias=False
        )  

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        # self.norms = []
        self.dropouts = []

        for _ in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters, self.filters))
            self.activations.append(Activation(act_func))
            # self.norms.append(nn.BatchNorm1d(self.filters))#et pas self.filters
            self.dropouts.append(nn.Dropout(p=prob_dropout))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        # self.norms = nn.ModuleList(self.norms)
        self.dropouts = nn.ModuleList(self.dropouts)
        return

    def forward(self, x):
        x = self.initial_activation(
            self.initial_layer(x)
        )  
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
            # seq = self.norms[i](seq)
        return self.output_layer(x)
