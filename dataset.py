from gflownet.envs.base import make_train_set, make_test_set

class Dataset:
    """
    Will load the dataset scored by the oracle
    """

    def __init__(self, config, env):
        self.config = config
        self.env = env
    
    def initialise_dataset(self):
        """
        Initialise the dataset by calling env specific functions
        Also initialise the dataset stats
        """
    
        self.dataset = self.env.make_train_set()

    def update_dataset(self, **kwargs):
        """
        Update the dataset with new data
        Also update the dataset stats
        """
        pass

    def get_dataloader(self):
        """
        Build and return the dataloader for the networks
        """
        pass