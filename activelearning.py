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
        #util class to handle the transition format  
        self.transition = FormatHandler()
        #util class to handle the statistics during training
        self.statistics = Statistics()

        #load the main components of the AL pipeline
        self.oracle = Oracle(self.config, self.transition)
        self.proxy = Proxy(self.config, self.transition)
        self.acq = AcquisitionFunction(self.config, self.proxy)
        self.env = Env(self.config, self.acq)
        self.gflownet = GFlowNet(self.config, self.env)
        self.querier = Querier(self.config, self.gflownet)
    
    def runPipeline(self):
        #useful variables
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
        '''
        Creates the working directories to store the data.
        It can call a class DataHandler which methods create the directories (if it is too long to put in this class)
        '''
        return

class FormatHandler():
    '''
    Contains the transition methods between the format of data in the pipeline. Can be passed to proxy, oracle, Gfn to handle the good data format before calling the networks
    '''
    def __init__(self) -> None:
        pass

class Statistics():
    '''
    Utils functions to compute and handle the statistics (saving them or send to comet).
    Incorporates the previous function "getModelState", ...
    Like FormatHandler, it can be passed on to querier, gfn, proxy, ... to get the statistics of training of the generated data at real time
    '''
    def __init__(self) -> None:
        pass

    def init_comet(self):
        return
    





'''

WHAT FOLLOWS IS THE SKELETON, SEE IN THE RELEVANT FOLDER FOR DETAILS

'''   
# #---------------------------------------------
# '''
# ORACLE
# '''

# class Oracle:
#     '''
#     Generic Class for the oracle. 
#     The different oracles (can be classes (MLP-toy oracle eg) or just a single function calling another annex program)
#     can be called according to a config param in the method score
#     '''
#     def __init__(self, config, transition):
#         self.config = config
#         self.transition = transition

#     def initializeDataset(self):
#         return
    
#     def score(self, queries):
#         '''
#         Calls the specific oracle (class/function) and apply its "score" method on the dataset
#         '''
#         #the transition.base2oracle method will be called to transform the queries in the input format of the oracle
#         return

#     def updateDataset(self, queries, energies):
#         return


# #---------------------------------------------
# '''
# PROXY
# '''
# class Proxy:
#     def __init__(self, config, transition):
#         self.config = config
#         self.transition = transition
#         self.init_model()
    
#     def init_model(self):
#         '''
#         Initialize the proxy we want (cf config). Each possible proxy is a class (MLP, transformer ...)
#         Ensemble methods will be another separate class
#         '''
#         return
    
#     def load_model(self):
#         '''
#         will not have to be used normally because the global object ActiveLearning.proxy will be continuously updated
#         '''
#         return
    
#     def converge(self):
#         '''
#         will call getDataLoaders/train / test / checkConvergencce 
#         '''
#         return
    
#     def train(self):
#         ''' 
#         will call getLoss
#         '''
#         return
    
#     def test(self):
#         return
    
#     def getLoss(self):
#         return
    
#     def checkConvergence(self):
#         return
    
#     def getDataLoaders(self):
#         '''
#         will instantiate the training dataset and shuffle/work on it
#         '''
#         dataset = BuildDataset(self.config, self.transition)
#         return
    
#     def evaluate(self):
#         return
    
# class BuildDataset:
#     '''
#     Will load the dataset scored by the oracle and convert it in the right format for the proxy with transition.base2proxy
#     '''
#     def __init__(self, config, transition):
#         self.config = config
#         self.transition = transition


# #----------------------------------------
# '''
# ACQUISITION FUNCTION
# '''
       
# class AcquisitionFunction:
#     '''
#     Cf Oracle class : generic AF class which calls the right AF sub_class
#     '''
#     def __init__(self, config, proxy):
#         self.config = config
#         self.proxy = proxy
        
#         #the specific class of the exact AF is instantiated here
    
#     def load_best_proxy(self):
#         '''
#         In case, loads the latest version of the proxy (no need normally)
#         '''
#         return
    
#     def get_reward_batch(self):
#         '''
#         calls the get_reward method of the appropriate Acquisition Class (MUtual Information, Expected Improvement, ...)
#         '''


# #----------------------------------------
# '''
# ENVIRONMENT
# '''

# class Env:
#     def __init__(self, config, acq):
#         self.config = config
#         self.acq = acq

#         self.init_env()

#     def init_env(self, idx = 0):
#         #should we initialize in terms of state rather  ? self.state = ([], None)
#         self.seq = []
#         self.fid = None
#         self.done = False
#         self.idx = idx
    
#     def get_action_space(self):
#         '''
#         get all possible actions to get the parents
#         '''
#         return
    
#     def get_mask(self):
#         '''
#         for sampling in GFlownet and masking in the loss function
#         '''
#         return
    
#     def get_parents(self):
#         '''
#         to build the training batch (for the inflows)
#         '''
#         return
    
#     def step(self):
#         '''
#         for forward sampling
#         '''
    
#     def acq2rewards(self):
#         '''
#         correction of the value of the AF for positive reward (or to scale it)
#         '''

#         return
    
#     def get_reward(self):
#         '''
#         get the reward values of a batch of candidates
#         '''


# #-------------------------------
# '''
# GFLOWNET
# '''

# class GFlowNet:
#     def __init__(self, config, env):
#         self.config = config
#         self.env = env

#         #set loss function, device, buffer ...
    
#     def init_model(self, load_best_model = True):
#         '''
#         Initializes the GFN policy network (separate class for MLP for now), and load the best one (random if not best GFN yet)
#         '''
        
#         return
    
#     def get_training_data(self):
#         '''
#         Calls the buffer to get some interesting training data
#         Performs backward sampling for off policy data and forward sampling
#         Calls the utils method forward sampling and backward sampling
#         '''
#         return
    
#     def forward_sampling(self):
#         return
    
#     def backward_sampling(self):
#         return
    
#     def flowmatch_loss(self):
#         return
    
#     def trajectory_balance(self):
#         return
    
#     def train(self):
#         return
    
#     def sample(self):
#         '''
#         Just performs forward sampling with the trained GFlownet
#         '''
#         return
 
# class Buffer:
#     '''
#     BUffer of data : 
#     - loads the data from oracle and put the best ones as offline training data
#     - maintains a replay buffer composed of the best trajectories sampled for training
#     '''
#     def __init__(self) -> None:
#         pass


# #-------------------------------
# '''
# QUERIER
# '''

# class Querier:
#     '''
#     Samples with the GFlownet latest model and then do post post processing to enhance the candidates and get statistics on them
#     '''
#     def __init__(self, config, gflownet):
#         self.config = config
#         self.gflownet = gflownet
    
#     def load_model(self):
#         '''
#         loads the best GFlownet (no need normally)
#         '''
#         return
    
#     def buildQuery(self):
#         '''
#         calls gflownet.sample() through sampleForQuery and then do post processing on it
#         '''
#         return
    
#     def sample4query(self):
#         return
    
#     def enhance_queries(self):
#         '''
#         runs filtering, annealing, ...
#         '''
#         return
    
#     def construct_query(self):
#         '''
#         Final filtering with fancy tools : clustering ...
#         '''
#         return