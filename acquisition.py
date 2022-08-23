class AcquisitionFunction:
    '''
    Cf Oracle class : generic AF class which calls the right AF sub_class
    '''
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        
        #the specific class of the exact AF is instantiated here
    
    def load_best_proxy(self):
        '''
        In case, loads the latest version of the proxy (no need normally)
        '''
        return
    
    def get_reward_batch(self):
        '''
        calls the get_reward method of the appropriate Acquisition Class (MUtual Information, Expected Improvement, ...)
        '''