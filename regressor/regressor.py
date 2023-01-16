import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import Adam
import hydra
from tqdm import tqdm


class DropoutRegressor:
    def __init__(
        self,
        device,
        checkpoint,
        training,
        dataset,
        config_model,
        config_env,
        logger=None,
    ):
        """
        Args:
            config specific to the surrogate model
            dataset class which has function to get dataloaders
            logger

        Inialises model and optimiser. Fits the model and saves it once convergence is reached.
        """
        self.logger = logger
        self.config_model = config_model
        self.config_env = config_env

        self.device = device

        # Training Parameters
        self.training_eps = training.eps
        self.max_epochs = training.max_epochs
        self.history = training.history
        assert self.history <= self.max_epochs
        self.batch_size = training.training_batch
        self.learning_rate = training.learning_rate
        self.weight_decay = training.weight_decay

        # Dataset
        self.dataset = dataset

        # Logger
        self.progress = self.logger.progress
        if checkpoint:
            self.logger.set_proxy_path(checkpoint)

    def init_model(self):
        """
        Initialize the network (MLP, Transformer, RNN)
        """
        self.model = hydra.utils.instantiate(
            self.config_model,
            config_env=self.config_env,
            _recursive_=False,
        ).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def load_model(self):
        """
        Load and returns the model
        """
        stem = (
            self.logger.proxy_ckpt_path.stem + self.logger.context + "final" + ".ckpt"
        )
        path = self.logger.proxy_ckpt_path.parent / stem

        self.init_model()

        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.to(self.device)
            for state in self.optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            return True
        else:
            raise FileNotFoundError

    def fit(self):
        """
        Initialises the model and dataloaders.
        Trains the model and saves it once convergence is attained.
        """
        # we reset the model, cf primacy bias, here we train on more and more data
        self.init_model()

        # for statistics we save the tr and te errors
        [self.err_tr_hist, self.err_te_hist] = [[], []]

        # get training data in torch format
        train_loader, test_loader = self.dataset.get_dataloader()

        pbar = tqdm(range(1, self.max_epochs + 1), disable=not self.progress)
        self.converged = 0

        for epoch in pbar:
            self.test(test_loader)

            # if model is the best so far
            self.logger.save_proxy(self.model, self.optimizer, final=False, epoch=epoch)

            self.train(train_loader)

            # after training at least "history" epochs, check convergence
            if epoch > self.history:
                self.check_convergence(epoch)
                if self.converged == 1:
                    self.logger.save_proxy(
                        self.model, self.optimizer, final=True, epoch=epoch
                    )
                    if self.progress:
                        print(
                            "Convergence reached in {} epochs with MSE {:.4f}".format(
                                epoch, self.err_te_hist[-1]
                            )
                        )
                    break

            if self.progress:
                description = "Train MSE: {:.4f} | Test MSE: {:.4f}".format(
                    self.err_tr_hist[-1], self.err_te_hist[-1]
                )
                pbar.set_description(description)

    def train(self, tr):
        """
        Args:
            train-loader
        """
        err_tr = []
        self.model.train(True)
        for x_batch, y_batch in tqdm(tr, disable=True):
            output = self.model(x_batch.to(self.device))
            loss = F.mse_loss(output[:, 0], y_batch.float().to(self.device))
            if self.logger:
                self.logger.log_metric("proxy_train_mse", loss.item())
            err_tr.append(loss.data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())

    def test(self, te):
        err_te = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in tqdm(te, disable=True):
                output = self.model(x_batch.to(self.device))
                loss = F.mse_loss(output[:, 0], y_batch.float().to(self.device))
                if self.logger:
                    self.logger.log_metric("proxy_val_mse", loss.item())
                err_te.append(loss.data)
        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())

    def check_convergence(self, epoch):
        eps = self.training_eps
        history = self.history
        max_epochs = self.max_epochs

        if all(
            np.asarray(self.err_te_hist[-history + 1 :]) > self.err_te_hist[-history]
        ):  # early stopping
            self.converged = 1  # not a legitimate criteria to stop convergence ...
            print("\nTest loss increasing.")

        if (
            abs(self.err_te_hist[-history] - np.average(self.err_te_hist[-history:]))
            / self.err_te_hist[-history]
            < eps
        ):
            self.converged = 1
            if self.progress:
                print("\nHit test loss convergence criterion.")

        if epoch >= max_epochs:
            self.converged = 1
            if self.progress:
                print("\nReached max_epochs.")

    def forward_with_uncertainty(self, x, num_dropout_samples=10):
        self.model.train()
        with torch.no_grad():
            outputs = torch.hstack(
                [self.model(x.to(self.device)) for _ in range(num_dropout_samples)]
            )
        return outputs
