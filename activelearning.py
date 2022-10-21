"""
Active learning with GFlowNets
"""

from omegaconf import DictConfig, OmegaConf
from oracle import Oracle
from proxy import Proxy
from acquisition import AcquisitionFunction
from env import Env
from gflownet import GFlowNet
from querier import Querier
from utils.logger import Logger
import wandb


# TODO : instantiate the config with hydra . Once we have the config object we can pass
# it to other functions
class ActiveLearning:
    """
    AL Global pipeline : so far it is a class (the config is easily passed on to all
    methods), very simple, but it can be a single simple function
    """

    def __init__(self, config):
        # assume we have a config object that we can pass to all the components of the
        # pipeline like in the previous code, eg "config_test.yaml"
        self.config = OmegaConf.load(config)
        # setup function that creates the directories to save data, ...
        self.setup()
        # util class to handle the statistics during training
        self.logger = Logger(self.config)
        # load the main components of the AL pipeline
        self.oracle = Oracle(self.config, self.logger)
        self.proxy = Proxy(self.config, self.logger)
        self.acq = AcquisitionFunction(self.config, self.proxy)
        self.env = Env(self.config, self.acq)
        self.gflownet = GFlowNet(self.config, self.logger, self.env)
        self.querier = Querier(self.config, self.gflownet)

    def run_pipeline(self):
        self.iter = None
        # we initialize the first dataset
        self.oracle.initialize_dataset()
        # we run each round of active learning
        for self.iter in range(self.config.al.n_iter):
            self.logger.set_context("iter{}".format(self.iter + 1))
            self.iterate()
        self.logger.finish()

    def iterate(self):
        self.proxy.train()
        self.gflownet.train()
        queries = self.querier.build_query()
        print(queries)
        energies = self.oracle.score(queries)
        self.oracle.update_dataset(queries, energies)

    def setup(self):
        """
        Creates the working directories to store the data.
        """
        return


if __name__ == "__main__":
    # TODO : activelearning pipeline as a simple function, without the class ?
    config_test_name = "config_test.yaml"

    al = ActiveLearning(config=config_test_name)
    al.run_pipeline()
