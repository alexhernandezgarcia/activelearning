import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import math
import copy
import os
from abc import abstractmethod
from torch.optim import Adam

class Model(nn.Module):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.device = self.config.device
        self.path_data = self.config.path.data_oracle
        self.path_model = self.config.path.model_proxy

        # Training Parameters
        self.training_eps = self.config.proxy.training.eps
        self.max_epochs = self.config.proxy.training.max_epochs
        self.history = self.config.proxy.training.history
        assert self.history <= self.max_epochs
        self.dropout = self.config.proxy.training.dropout
        self.batch_size = self.config.proxy.training.training_batch

        # Dataset management
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed

        self.model_class = NotImplemented  # will be precised in child classes

    @abstractmethod
    def init_model(self):
        """
        Initialize the proxy we want (cf config). Each possible proxy is a class (MLP, transformer ...)
        Ensemble methods will be another separate class
        """
        self.model = self.model_class(self.config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)

    @abstractmethod
    def load_model(self, dir_name=None):
        """
        will not have to be used normally because the global object ActiveLearning.proxy will be continuously updated
        """
        if dir_name == None:
            dir_name = self.config.path.model_proxy

        self.init_model()

        if os.path.exists(dir_name):
            checkpoint = torch.load(dir_name)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.device == "cuda":
                self.model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        # TODO: return self.model so that its easily integrated with proxy model
        else:
            raise NotImplementedError

    @abstractmethod
    def converge(self, data_handler):
        """
        will call getDataLoaders/train_batch/ test / checkConvergencce
        """
        # we reset the model, cf primacy bias, here we train on more and more data
        self.init_model()

        # for statistics we save the tr and te errors
        [self.err_tr_hist, self.err_te_hist] = [[], []]

        # get training data in torch format
        tr, te = data_handler.get_data_loaders()

        self.converged = 0
        self.epochs = 0

        while self.converged != 1:

            if (
                self.epochs > 0
            ):  #  this allows us to keep the previous model if it is better than any produced on this run
                self.train(tr)  # already appends to self.err_tr_hist
            else:
                self.err_tr_hist.append(0)

            self.test(te)

            if self.err_te_hist[-1] == np.min(
                self.err_te_hist
            ):  # if this is the best test loss we've seen
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.path_model,
                )  # we update only the best, not keep the previous ones
                print(
                    "new best found at epoch {}, of value {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )

            # after training at least "history" epochs, check convergence
            if self.epochs >= self.history + 1:
                self.check_convergence()

            if (self.epochs % 10 == 0) and self.config.debug:
                print(
                    "Model epoch {} test loss {:.4f}".format(
                        self.epochs, self.err_te_hist[-1]
                    )
                )

            self.epochs += 1

            # TODO : implement comet logger (with logger object in activelearning.py)
            # if self.converged == 1:
            #     self.statistics.log_comet_proxy_training(
            #         self.err_tr_hist, self.err_te_hist
            #     )

    @abstractmethod
    def train(self, tr):
        """
        will call getLoss
        """
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.get_loss(trainData)
            self.logger.log_metric("proxy_train_mse", loss.item())
            err_tr.append(loss.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.err_te_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())

    @abstractmethod
    def test(self, te):
        err_te = []
        self.model.eval()
        with torch.no_grad():
            for i, testData in enumerate(te):
                loss = self.get_loss(testData)
                self.logger.log_metric("proxy_val_mse", loss.item())
                err_te.append(loss.data)

        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())

    @abstractmethod
    def get_loss(self, data):
        inputs = data[0]
        targets = data[1]
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        input_lens = data[2]
        inputs = inputs.to(self.device)
        input_lens = input_lens.to(self.device)
        targets = targets.to(self.device)
        output = self.model(inputs, input_lens, targets)
        loss = F.mse_loss(output[:, 0], targets.float())
        return loss

    @abstractmethod
    def check_convergence(self):
        eps = self.training_eps
        history = self.history
        max_epochs = self.max_epochs

        if all(
            np.asarray(self.err_te_hist[-history + 1 :]) > self.err_te_hist[-history]
        ):  # early stopping
            self.converged = 1  # not a legitimate criteria to stop convergence ...
            print(
                "Model converged after {} epochs - test loss increasing at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )

        if (
            abs(self.err_te_hist[-history] - np.average(self.err_te_hist[-history:]))
            / self.err_te_hist[-history]
            < eps
        ):
            self.converged = 1
            print(
                "Model converged after {} epochs - hit test loss convergence criterion at {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )

        if self.epochs >= max_epochs:
            self.converged = 1
            print(
                "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(
                    self.epochs + 1, min(self.err_te_hist)
                )
            )

    @abstractmethod
    def evaluate(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data).cpu().detach().numpy()
            return output
