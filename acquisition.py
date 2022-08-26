import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
'''
ACQUISITION WRAPPER
'''

class AcquisitionFunction:
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy.proxy

        self.init_acquisition()
    
    def init_acquisition(self):
        if self.config.acquisition.main == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        else:
            raise NotImplementedError
    
    def get_reward(self, inputs_af_base):
        outputs = self.acq.get_reward_batch(inputs_af_base)
        return  outputs
    

'''
BASE CLASS FOR ACQUISITION
'''
class AcquisitionFunctionBase:
    '''
    Cf Oracle class : generic AF class which calls the right AF sub_class
    '''
    @abstractmethod
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.device = self.config.device
        
        #the specific class of the exact AF is instantiated here
    @abstractmethod
    def load_best_proxy(self):
        '''
        In case, loads the latest version of the proxy (no need normally)
        '''
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)
        
        else:
            raise FileNotFoundError

    @abstractmethod
    def get_reward_batch(self, inputs_af_base):
        '''
        calls the get_reward method of the appropriate Acquisition Class (MUtual Information, Expected Improvement, ...)
        '''
        pass

'''
SUBCLASS SPECIFIC ACQUISITION
'''

class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)
    
    def load_best_proxy(self):
        super().load_best_proxy()

    
    def get_reward_batch(self, inputs_af_base): #inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)

        inputs_af = list(map(self.base2af, inputs_af_base))
        inputs = torch.stack(inputs_af).view(len(inputs_af_base), -1)
    
        self.load_best_proxy()
        self.proxy.model.eval()
        with torch.no_grad():
            outputs = self.proxy.model(inputs)
        return outputs.cpu().detach()


    
    def base2af(self, state):
        #useful format
        self.dict_size = self.config.env.dict_size
        self.min_len = self.config.env.min_len
        self.max_len = self.config.env.max_len

        seq = state
        initial_len = len(seq)
        #into a tensor and then ohe
        seq_tensor = torch.from_numpy(seq)
        seq_ohe = F.one_hot(seq_tensor.long(), num_classes = self.dict_size +1)
        seq_ohe = seq_ohe.reshape(1, -1).float()
        #addind eos token
        eos_tensor = torch.tensor([self.dict_size])
        eos_ohe = F.one_hot(eos_tensor.long(), num_classes=self.dict_size + 1)
        eos_ohe = eos_ohe.reshape(1, -1).float()

        input_proxy = torch.cat((seq_ohe, eos_ohe), dim = 1)
        #adding 0-padding
        number_pads = self.max_len - initial_len
        if number_pads:
            padding = torch.cat(
                [torch.tensor([0] * (self.dict_size +1))] * number_pads
            ).view(1, -1)
            input_proxy = torch.cat((input_proxy, padding), dim = 1)
        
        return input_proxy.to(self.device)[0]
    


