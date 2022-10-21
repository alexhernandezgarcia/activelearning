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

    def __init__(self, args):
        self.run = wandb.init(
            config=args, project="ActiveLearningPipeline", name="test"
        )
        self.context = ""

    def set_context(self, context):
        self.context = context

    def log_metric(self, key, value, use_context=True):
        if use_context:
            key = self.context + "/" + key
        wandb.log({key: value})

    def log_histogram(self, key, value, use_context=True):
        if use_context:
            key = self.context + "/" + key
        plt.hist(value)
        plt.title(key)
        plt.ylabel("Frequency")
        plt.xlabel(key)
        fig = wandb.Image(plt)
        wandb.log({key: fig})
        # wandb.log({key: wandb.Histogram(value)})

    def finish(self):
        wandb.finish()
