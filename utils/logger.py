"""
Logger utils, to be developed.
"""
import wandb
import os
import torch
import matplotlib.pyplot as plt

class Logger:
    """
    Utils functions to compute and handle the statistics (saving them or send to
    comet).  Incorporates the previous function "getModelState", ...  Like
    FormatHandler, it can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """
    def __init__(self, config):
        self.config = config
        run_name = "proxy{}_oracle{}_gfn{}_minLen{}_maxLen{}".format(config.proxy.model.upper(), config.oracle.main.upper(), config.gflownet.policy_model.upper(), config.env.min_len, config.env.max_len)
        self.run = wandb.init(config=config, project='ActiveLearningPipeline', name=run_name)
        self.context = ""

    def set_context(self, context):
        self.context = context

    def log_metric(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        wandb.log({key:value})
    
    def log_histogram(self, key, value, use_context=True):
        if use_context:
            key = self.context + '/' + key
        fig = plt.figure()
        plt.hist(value)
        plt.title(key)
        plt.ylabel('Frequency')
        plt.xlabel(key)
        fig = wandb.Image(fig)
        wandb.log({key: fig})
        # wandb.log({key: wandb.Histogram(value)})
    
    def finish(self):
        wandb.finish()

