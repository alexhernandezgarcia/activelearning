
'''
ACQUISITION WRAPPER
'''
class AcquisitionFunction:
    def __init__(self, config, proxy):
        self.config = config
        self.proxy_wrapper = proxy
        self.proxy = self.proxy_wrapper.proxy

        self.init_acquisition()
    
    def init_acquisition(self):
        if self.config.acquisition.main == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        else:
            raise NotImplementedError
    

'''
BASE CLASS FOR ACQUISITION
'''
class AcquisitionFunctionBase:
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

'''
SUBCLASS SPECIFIC ACQUISITION
'''

class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
