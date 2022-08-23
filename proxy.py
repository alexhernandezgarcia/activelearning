class Proxy:
    def __init__(self, config, transition):
        self.config = config
        self.transition = transition
        self.init_model()
    
    def init_model(self):
        '''
        Initialize the proxy we want (cf config). Each possible proxy is a class (MLP, transformer ...)
        Ensemble methods will be another separate class
        '''
        return
    
    def load_model(self):
        '''
        will not have to be used normally because the global object ActiveLearning.proxy will be continuously updated
        '''
        return
    
    def converge(self):
        '''
        will call getDataLoaders/train / test / checkConvergencce 
        '''
        return
    
    def train(self):
        ''' 
        will call getLoss
        '''
        return
    
    def test(self):
        return
    
    def getLoss(self):
        return
    
    def checkConvergence(self):
        return
    
    def getDataLoaders(self):
        '''
        will instantiate the training dataset and shuffle/work on it
        '''
        dataset = BuildDataset(self.config, self.transition)
        return
    
    def evaluate(self):
        return
    
class BuildDataset:
    '''
    Will load the dataset scored by the oracle and convert it in the right format for the proxy with transition.base2proxy
    '''
    def __init__(self, config, transition):
        self.config = config
        self.transition = transition