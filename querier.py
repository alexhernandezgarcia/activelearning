class Querier:
    '''
    Samples with the GFlownet latest model and then do post post processing to enhance the candidates and get statistics on them
    '''
    def __init__(self, config, gflownet):
        self.config = config
        self.gflownet = gflownet.gflownet
        self.n_queries = self.config.al.queries_per_it

    
    def build_query(self):
        '''
        calls gflownet.sample() through sampleForQuery and then do post processing on it
        '''
        queries = self.sample4query()
        queries = self.enhance_queries(queries)
        queries = self.construct_query(queries)
        return queries
    
    def sample4query(self):
        self.gflownet.sample_sequence(self.n_queries)
        return
  
    def enhance_queries(self, queries):
        '''
        runs filtering, annealing, ...
        '''
        return queries

    def construct_query(self, queries):
        '''
        Final filtering with fancy tools : clustering ...
        '''
        return queries