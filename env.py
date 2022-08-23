class Env:
    def __init__(self, config, acq):
        self.config = config
        self.acq = acq

        self.init_env()

    def init_env(self, idx = 0):
        #should we initialize in terms of state rather  ? self.state = ([], None)
        self.seq = []
        self.fid = None
        self.done = False
        self.idx = idx
    
    def get_action_space(self):
        '''
        get all possible actions to get the parents
        '''
        return
    
    def get_mask(self):
        '''
        for sampling in GFlownet and masking in the loss function
        '''
        return
    
    def get_parents(self):
        '''
        to build the training batch (for the inflows)
        '''
        return
    
    def step(self):
        '''
        for forward sampling
        '''
    
    def acq2rewards(self):
        '''
        correction of the value of the AF for positive reward (or to scale it)
        '''

        return
    
    def get_reward(self):
        '''
        get the reward values of a batch of candidates
        '''
