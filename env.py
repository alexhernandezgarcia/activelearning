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

        self.device = self.config.device


    @abstractmethod
    def create_new_env(self, idx):
        pass

    @abstractmethod
    def init_env(self, idx):
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
    def get_parents(self, backward = False):
        '''
        to build the training batch (for the inflows)
        '''
        pass
    
    @abstractmethod
    def step(self,action):
        '''
        for forward sampling
        '''
        pass
    
    @abstractmethod
    def acq2rewards(self, acq_values):
        '''
        correction of the value of the AF for positive reward (or to scale it)
        '''

        pass
    
    @abstractmethod
    def get_reward(self, states, done):
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

        self.action_space = self.get_action_space()
        self.token_eos = self.get_token_eos(self.action_space)

        self.env_class = EnvAptamers

        self.init_env()
   
    def create_new_env(self, idx):
        env = EnvAptamers(self.config, self.acq)
        env.init_env(idx)
        return env
    
    def init_env(self, idx=0):
        super().init_env(idx)
        self.state = np.array([])
        self.n_actions_taken = 0
        self.done = False
        self.id = idx
        self.last_action = None

    def get_action_space(self):
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
        
        seq_len = len(self.state)

        if seq_len < self.min_seq_len:
            mask[self.token_eos] = 0
            return mask
        
        elif seq_len == self.max_seq_len:
            mask[:self.token_eos] = [0] * len(self.action_space)
            return mask
        
        else:
            return mask
    
    def get_parents(self, backward = False):
        super().get_parents(backward)

        if self.done:
            if self.state[-1] == self.token_eos:
                parents_a = [self.token_eos]
                parents = [self.state[:-1]]
                if backward:
                    self.done = False
                return parents, parents_a 
            else:
                raise NameError
        
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if self.state[-len(a): ] == list(a):
                    parents.append((self.state[:-len(a)]))
                    actions.append(idx)
            
            return parents, actions
    
    def step(self, action):
        super().step(action)
        valid = False
        seq = self.state
        seq_len = len(seq)
       
        if (action == [self.token_eos]) and (self.done == False):
            if seq_len >= self.min_seq_len and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                self.done = True
                self.n_actions_taken += 1
                self.state = next_seq
                self.last_action = self.token_eos

                return next_seq, action, valid
        
        if self.done == True:
            valid = False
            return None, None, valid
        
        elif self.done == False and not(action == [self.token_eos]):
            if action in list(map(list, self.action_space)) and seq_len <= self.max_seq_len:
                valid = True
                next_seq = np.append(seq, action)
                self.n_actions_taken += 1
                self.state = next_seq
                self.last_action = action
                return next_seq, action, valid
        
        else:
            raise TypeError("invalid action to take")
    
    def acq2reward(self, acq_values):
        min_reward = 1e-10
        true_reward = np.clip(acq_values, min_reward, None)
        customed_af = lambda x: x**3 #to favor the higher rewards in a more spiky way, can be customed
        exponentiate = np.vectorize(customed_af)
        return exponentiate(true_reward)

    def get_reward(self, states, done):
        rewards = np.zeros(len(done), dtype = float)
        final_states = [s for s, d in zip(states, done) if d]
        inputs_af_base = [self.manip2base(final_state) for final_state in final_states]
        
        final_rewards = self.acq.get_reward(inputs_af_base).view(len(final_states)).numpy()
        final_rewards = self.acq2reward(final_rewards)

        done = np.array(done)
        
        rewards[done] = final_rewards
        return rewards
        
    def base2manip(self, state):
        seq_base = state
        seq_manip = np.concatenate((seq_base, [self.token_eos]))
        return seq_manip
    
    def manip2base(self, state):
        seq_manip = state
        if seq_manip[-1] == self.token_eos:
            seq_base = seq_manip[:-1]
            return seq_base
        else:
            raise TypeError

    
