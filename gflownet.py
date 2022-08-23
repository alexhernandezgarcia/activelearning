class GFlowNet:
    def __init__(self, config, env):
        self.config = config
        self.env = env

        #set loss function, device, buffer ...
    
    def init_model(self, load_best_model = True):
        '''
        Initializes the GFN policy network (separate class for MLP for now), and load the best one (random if not best GFN yet)
        '''
        
        return
    
    def get_training_data(self):
        '''
        Calls the buffer to get some interesting training data
        Performs backward sampling for off policy data and forward sampling
        Calls the utils method forward sampling and backward sampling
        '''
        return
    
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
 
class Buffer:
    '''
    BUffer of data : 
    - loads the data from oracle and put the best ones as offline training data
    - maintains a replay buffer composed of the best trajectories sampled for training
    '''
    def __init__(self) -> None:
        pass