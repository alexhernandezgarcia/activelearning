#To load the config
from omegaconf import DictConfig, OmegaConf
import hydra

#Import main components of the pipeline
from oracle import Oracle
from proxy import Proxy
from acquisition import AcquisitionFunction
from env import Env
from gflownet import GFlowNet
from querier import Querier

#--------------------------------
'''
AL Global pipeline
'''

#TODO : instantiate the config with hydra . Once we have the config object we can pass it to other functions
class ActiveLearning:
    def __init__(self, config):
        #assume we have a config object that we can pass to all the components of the pipeline like in the previous code, eg "config_test.yaml"
        self.config = OmegaConf.load(config)
        
        #setup function that creates the directories to save data, ...
        self.setup()
        #util class to handle the statistics during training
        self.logger = Logger()

        #load the main components of the AL pipeline
        self.oracle = Oracle(self.config)
        self.proxy = Proxy(self.config, self.logger)
        self.acq = AcquisitionFunction(self.config, self.proxy)
        self.env = Env(self.config, self.acq)
        self.gflownet = GFlowNet(self.config, self.logger, self.env)
        self.querier = Querier(self.config, self.gflownet)
    
    def run_pipeline(self):

        self.iter = None

        #we initialize the first dataset
        self.oracle.initialize_dataset()
        
        #we run each round of active learning
        for self.iter in range(self.config.al.n_iter):
            self.iterate()

    
    def iterate(self):
        self.proxy.train()
        
        self.gflownet.train()
        
        queries = self.querier.build_query()
        print(queries)
        energies = self.oracle.score(queries)
        self.oracle.update_dataset(queries, energies)
    
    def setup(self):
        '''
        Creates the working directories to store the data.
        It can call a class DataHandler which methods create the directories (if it is too long to put in this class)
        '''
        return

class Logger():
    '''
    Utils functions to compute and handle the statistics (saving them or send to comet).
    Incorporates the previous function "getModelState", ...
    Like FormatHandler, it can be passed on to querier, gfn, proxy, ... to get the statistics of training of the generated data at real time
    '''
    def __init__(self):
        pass

    def init_comet(self):
        return

if __name__ == "__main__":
    config_test_name = "config_test.yaml"
    al = ActiveLearning(config = config_test_name)
    al.run_pipeline()

    config = OmegaConf.load("config_test.yaml")


