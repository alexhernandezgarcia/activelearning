class Querier:
    '''
    Samples with the GFlownet latest model and then do post post processing to enhance the candidates and get statistics on them
    '''
    def __init__(self, config, gflownet):
        self.config = config
        self.gflownet = gflownet
    
    def load_model(self):
        '''
        loads the best GFlownet (no need normally)
        '''
        return
    
    def buildQuery(self):
        '''
        calls gflownet.sample() through sampleForQuery and then do post processing on it
        '''
        return
    
    def sample4query(self):
        return
    
    def enhance_queries(self):
        '''
        runs filtering, annealing, ...
        '''
        return
    
    def construct_query(self):
        '''
        Final filtering with fancy tools : clustering ...
        '''
        return