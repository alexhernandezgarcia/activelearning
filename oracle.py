class Oracle:
    '''
    Generic Class for the oracle. 
    The different oracles (can be classes (MLP-toy oracle eg) or just a single function calling another annex program)
    can be called according to a config param in the method score
    '''
    def __init__(self, config, transition):
        self.config = config
        self.transition = transition

    def initializeDataset(self):
        return
    
    def score(self, queries):
        '''
        Calls the specific oracle (class/function) and apply its "score" method on the dataset
        '''
        #the transition.base2oracle method will be called to transform the queries in the input format of the oracle
        return

    def updateDataset(self, queries, energies):
        return