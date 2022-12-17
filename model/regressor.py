import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import Adam
import hydra

ACTIVATION_KEY = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}


class DropoutRegressor(nn.Module):
    def __init__(self, config, config_network, config_env, dataset=None, logger=None):
        """
        Args:
            config specific to the surrogate model
            dataset class which has function to get dataloaders
            logger

        Inialises model and optimiser. Fits the model and saves it once convergence is reached.
        """
        self.config = config
        self.logger = logger
        self.config_network = config_network
        self.config_env = config_env

        self.device = self.config.device
        self.path_model = self.config.path_model

        # Training Parameters
        self.training_eps = self.config.training.eps
        self.max_epochs = self.config.training.max_epochs
        self.history = self.config.training.history
        assert self.history <= self.max_epochs
        self.batch_size = self.config.training.training_batch

        # Dataset management
        self.dataset = dataset
        self.shuffle_data = self.config.proxy.data.shuffle
        self.seed_data = self.config.proxy.data.seed

    def init_model(self):
        """
        Initialize the network (MLP, Transformer, RNN)
        """
        self.model = hydra.utils.instantiate(
            self.config_network, config_env=self.config_env
        ).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(), self.config.learning_rate, self.config.weight_decay
        )

    def load_model(self, dir_name=None):
        """
        Load and returns the model
        """
        if dir_name == None:
            dir_name = self.config.path.model_proxy

        self.init_model()

        if os.path.exists(dir_name):
            checkpoint = torch.load(dir_name)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.to(self.device)
            for state in self.optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            return self.model
        else:
            raise FileNotFoundError

    def save_model(self):
        """
        Saves model once convergence is attained.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.path_model,
        )

    def fit(self):
        """
        Initlaises the model and dataloaders.
        Trains the model and saves it once convergence is attained.
        """
        # we reset the model, cf primacy bias, here we train on more and more data
        self.init_model()

        # for statistics we save the tr and te errors
        [self.err_tr_hist, self.err_te_hist] = [[], []]

        # get training data in torch format
        train_loader, test_loader = self.dataset.get_data_loaders()

        self.converged = 0
        self.epochs = 0

        while self.converged != 1:

            if (
                self.epochs > 0
            ):  #  this allows us to keep the previous model if it is better than any produced on this run
                self.train(train_loader)  # already appends to self.err_tr_hist
            else:
                self.err_tr_hist.append(0)

            self.test(test_loader)
            if self.err_te_hist[-1] == np.min(
                self.err_te_hist
            ):  # if this is the best test loss we've seen
                self.save_model()
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

    def train(self, tr):
        """
        Args:
            train-loader
        """
        err_tr = []
        self.model.train(True)
        for x_batch, y_batch in enumerate(tr):
            output = self.model(x_batch)
            loss = F.mse_loss(output[:, 0], y_batch.float())
            self.logger.log_metric("proxy_train_mse", loss.item())
            err_tr.append(loss.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.err_te_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())

    def test(self, te):
        err_te = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in enumerate(te):
                output = self.model(x_batch)
                loss = F.mse_loss(output[:, 0], y_batch.float())
                self.logger.log_metric("proxy_val_mse", loss.item())
                err_te.append(loss.data)
        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())

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

    def forward_with_uncertainty(self, x):
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat(
                [self.forward(x) for _ in range(self.config.num_dropout_samples)]
            )
        return outputs.mean(dim=0), outputs.std(dim=0)
