from omegaconf import DictConfig, OmegaConf
import hydra

#TODO : instantiate the config with hydra . Once we have the config object we can pass it to other functions
class ActiveLearning:
    def __init__(self, config):
        self.config = OmegaConf.load(config)
        
        self.oracle = Oracle(self.config)
        self.proxy = Proxy(self.config)
        self.acq = AcquisitionFunction(self.config, self.proxy)
        self.env = Env(self.config, self.acq)
        self.gflownet = GFlowNet(self.config, self.env)
        self.querier = Querier(self.config, self.gflownet)

        #setup function that creates the directories to save data, ...
        self.setup()

    
    def runPipeline(self):
        self.iter = None
        self.terminal = 0

        #we initialize the first dataset
        self.oracle.initializeDataset()

        #we run each round of active learning
        for self.iter in range(self.config.al.n_iter):
            if self.iter == (self.config.al.n_iter - 1):
                self.terminal = 1
            self.iterate()
    
    def iterate(self):
        self.proxy.converge()
        self.gflownet.train()

        if self.terminal == 0:
            queries = self.querier.buildQuery()
            energies = self.oracle.score(queries)
            self.oracle.updateDataset(queries, energies)
    
    def setup(self):
        return








class Oracle:
    def __init__(self, config):
        self.config = config

    def initializeDataset(self):
        return
    
    def score(self, queries):
        return

    def updateDataset(self, queries, energies):
        return

class Proxy:
    def __init__(self, config):
        self.config = config
    
    def converge(self):
        return

class AcquisitionFunction:
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy

class Env:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq

class GFlowNet:
    def __init__(self, config, env):
        self.config = config
        self.env = env
    
    def train(self):
        return

class Querier:
    def __init__(self, config, gflownet):
        self.config = config
        self.gflownet = gflownet
    
    def buildQuery(self):
        return
