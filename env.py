import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import itertools
#to check path
import os
#for mother class of Oracle
from abc import abstractmethod


'''
ENV WRAPPER
'''

class Env:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq

        self.init_env()
    
    def init_env(self):
        if self.config.env.main == "aptamers":
            self.env = EnvAptamers(self.config, self.acq)
        else:
            raise NotImplementedError

    



'''
Generic Env Base Class
'''

class EnvBase:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq

    @abstractmethod
    def init_env(self, idx = 0):
        pass


    @abstractmethod
    def get_action_space(self):
        '''
        get all possible actions to get the parents
        '''
        pass
    
    @abstractmethod
    def get_mask(self):
        '''
        for sampling in GFlownet and masking in the loss function
        '''
        pass
    
    @abstractmethod
    def get_parents(self):
        '''
        to build the training batch (for the inflows)
        '''
        pass
    
    @abstractmethod
    def step(self):
        '''
        for forward sampling
        '''
        pass
    
    @abstractmethod
    def acq2rewards(self):
        '''
        correction of the value of the AF for positive reward (or to scale it)
        '''

        pass
    
    @abstractmethod
    def get_reward(self):
        '''
        get the reward values of a batch of candidates
        '''
        pass


'''
Specific Envs
'''

class EnvAptamers(EnvBase):
    def __init__(self, config, acq) -> None:
        super().__init__(config, acq)

        self.device = self.config.device

        self.max_seq_len = self.config.env.max_len
        self.min_seq_len = self.config.env.min_len
        self.max_word_len = self.config.env.max_word_len
        self.min_word_len = self.config.env.min_word_len
        self.n_alphabet = self.config.env.dict_size

        self.action_space = self.get_actions_space()
        self.token_eos = self.get_token_eos(self.action_space)

        self.env_class = EnvAptamers

        self.init_env()
    
    def init_env(self, idx=0):
        super().init_env(idx)
        self.seq = np.array([])
        self.n_actions_taken = 0
        self.done = False
        self.id = idx

    def get_actions_space(self):
        super().get_action_space()
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len +1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat = r)]
            actions += actions_r
        return actions

    def get_token_eos(self, action_space):
        return len(action_space)
    
    def get_mask(self):
        super().get_mask()

        mask = [1] * (len(self.action_space) + 1)

        if self.done : 
            return [0 for _ in mask]
        
        seq_len = len(self.seq)

        if seq_len < self.min_seq_len:
            mask[self.token_eos] = 0
            return mask
        
        elif seq_len == self.max_seq_len:
            mask[:self.token_eos] = [0] * len(self.action_space)
        
        else:
            return mask
    
    def get_parents(self, backward = False):
        super().get_parents()

        if self.done:
            if self.seq[-1] == self.token_eos:
                parents_a = [self.token_eos]
                parents = [self.seq[:-1]]
                if backward:
                    self.done = False
                return parents, parents_a 
            else:
                raise NameError
        
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if self.seq[-len(a): ] == list(a):
                    parents.append((self.seq[:-len(a)]))
                    actions.append(idx)
            
            return parents, actions
    
    def step(self, action):
        super().step()
        valid = False
        seq = self.seq
        seq_len = len(seq)

        if (action == self.token_eos) and (self.done == False):
            if seq_len >= self.min_seq_len and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                self.done = True
                self.n_actions_taken += 1
                self.seq = next_seq

                return next_seq, action, valid
        
        if self.done == True:
            valid = False
            return None, None, valid
        
        elif self.done == False and not(action == self.token_eos):
            if action in list(map(list, self.action_space)) and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                self.n_actions_taken += 1
                self.seq = next_seq
                return next_seq, action, valid
        
        else:
            raise TypeError("invalid action to take")
        
    def base2manip(self, state):
        return
    
    
