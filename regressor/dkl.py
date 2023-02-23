import torch
import gpytorch
import hydra
from gflownet.utils.common import set_device, set_float_precision
from utils.dkl_utils import batched_call
from gpytorch.settings import cholesky_jitter
from gpytorch.lazy import ConstantDiagLazyTensor
import math
from torch import LongTensor
from gpytorch import lazify
import numpy as np
from torch.nn import functional as F
from model.mlm import sample_mask
from gpytorch.variational import IndependentMultitaskVariationalStrategy
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from model.mlm import mlm_eval_epoch
from model.shared_elements import check_early_stopping
import copy
from torch.nn.utils.rnn import pad_sequence


class Tokenizer:
    def __init__(self, non_special_vocab):
        self.non_special_vocab = non_special_vocab
        # skipped UNK and 0
        self.special_vocab = ["[SEP]", "[CLS]", "[PAD]", "[MASK]"]
        self.full_vocab = self.non_special_vocab + self.special_vocab
        self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}
        # self.lookup["[PAD]"] = -2
        # self.inverse_lookup[-2] = "[PAD]"
        padding_token = "[PAD]"
        masking_token = "[MASK]"
        bos_token = "[CLS]"
        eos_token = "[SEP]"
        self.padding_idx = self.lookup[padding_token]
        assert self.padding_idx == 22
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[
            eos_token
        ]  # preferably ensure eos here is the same eos in gfn
        self.sampling_vocab = non_special_vocab
        self.special_idxs = [
            self.padding_idx,
            self.bos_idx,
            self.eos_idx,
            self.masking_idx,
        ]

    def transform(self, list_of_seq_tensors):
        # add [CLS] and [SEP] tokens to the beginning and end of each list in seq_list
        # seq_list is a list of tensors
        # return a list of lists
        seq_list = [
            torch.cat([torch.tensor([self.bos_idx]), seq, torch.tensor([self.eos_idx])])
            for seq in list_of_seq_tensors
        ]
        states = pad_sequence(
            seq_list,
            batch_first=True,
            padding_value=self.padding_idx,
        )

        # Add [CLS] and [SEP] tokens to the beginning and end of each sequence
        # cls_tensor = torch.ones(seq_array.shape[0], 1).long() * self.bos_idx
        # sep_tensor = torch.ones(seq_array.shape[0], 1).long() * self.eos_idx
        # cls_tensor = cls_tensor.to(seq_array)
        # sep_tensor = sep_tensor.to(seq_array)
        # seq_array = torch.cat([cls_tensor, seq_array, sep_tensor], dim=1)
        return states  # torch.Size([1280, 51])


class DeepKernelRegressor:
    def __init__(
        self,
        logger,
        device,
        dataset,
        config_model,
        surrogate,
        float_precision,
        tokenizer,
        encoder_obj="mlm",
        batch_size=32,
        **kwargs,
    ):
        self.logger = logger
        self.progress = self.logger.progress

        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

        # Dataset
        self.dataset = dataset
        # self.n_samples = dataset.n_samples

        self.tokenizer = tokenizer
        # config_model.model["tokenizer"] = self.tokenizer
        # self.encoder_config = encoder
        self.language_model = hydra.utils.instantiate(
            config_model,
            tokenizer=self.tokenizer,
            device=self.device,
            float_precision=self.float,
            _recursive_=False,
        )
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate = hydra.utils.instantiate(
            surrogate,
            tokenizer=self.language_model.tokenizer,
            encoder=self.language_model,
            device=self.device,
            float_precision=self.float,
        )

        self.batch_size = batch_size

    # def get_features(self, seq_array, batch_size=None, transform=True):
    #     # batch_size not used
    #     # if transform:
    #     #     original_shape = seq_array.shape
    #     #     flat_seq_array = seq_array.reshape(-1)
    #     # else:
    #     original_shape = seq_array.shape[:-1]
    #     flat_seq_array = seq_array.flatten(end_dim=-2)

    #     # Train transform = data augmentations + test transform

    #     # if self.training and transform:
    #     #     enc_seq_array = gfp_transforms.padding_collate_fn(
    #     #         [self.train_transform(seq) for seq in flat_seq_array],
    #     #         self.tokenizer.padding_idx,
    #     #     )
    #     # elif transform:  # transforms from string to int
    #     #     enc_seq_array = gfp_transforms.padding_collate_fn(
    #     #         [self.test_transform(seq) for seq in flat_seq_array],
    #     #         self.tokenizer.padding_idx,
    #     #     )
    #     # else:
    #     #     enc_seq_array = seq_array

    #     enc_seq_array = seq_array.to(
    #         self.device
    #     )  # torch.Size([32, 36]), padded states
    #     features = self.language_model(
    #         enc_seq_array
    #     )  # torch.Size([32, 16]) --> pooled features where we had considered 0s for both the padding and the EOS element, encoder here is the entire LanguageModel

    #     # TODO: what is original shape?
    #     return features.view(*original_shape, -1)

    def initialize_surrogate(self, X_train, Y_train):
        # Is X_train 32 or 426?
        # X_train = torch.Size([5119, 51])
        if (
            hasattr(self.surrogate, "init_inducing_points")
            and self.surrogate.num_inducing_points <= X_train.shape[0]
        ):
            print("\n---- initializing GP variational params ----")
            self.surrogate.eval()
            self.surrogate.requires_grad_(False)
            init_features = torch.cat(
                batched_call(self.surrogate.get_features, X_train, self.batch_size)
            )
            # try:
            self.surrogate.train()
            self.surrogate.init_inducing_points(
                init_features
            )  # botorch.SingleTaskVariationalGP based function
            self.surrogate.initialize_var_dist_sgpr(
                init_features,
                Y_train.to(init_features),
                noise_lb=1.0,
            )
            print("variational initialization successful")

        self.mll = VariationalELBO(
            self.surrogate.likelihood, self.surrogate.model, num_data=X_train.shape[0]
        )
        self.mll.to(self.surrogate.device)

    def mlm_train_step(self, optimizer, token_batch, mask_ratio, loss_scale=1.0):
        optimizer.zero_grad(set_to_none=True)

        # replace random tokens with mask token
        mask_idxs = sample_mask(token_batch, self.tokenizer, mask_ratio)  # (32, 5)
        masked_token_batch = token_batch.clone().to(self.language_model.device)
        np.put_along_axis(
            masked_token_batch, mask_idxs, self.tokenizer.masking_idx, axis=1
        )  # masking_idx = 4

        # get predicted logits for masked tokens
        logits, _ = self.language_model.logits_from_tokens(masked_token_batch)
        vocab_size = logits.shape[-1]
        masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(
            -1, vocab_size
        )  # torch.Size([160, 26])

        # use the ground-truth tokens as labels
        masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
        masked_tokens = masked_tokens.view(-1).to(
            self.language_model.device
        )  # torch.Size([160])

        loss = loss_scale * F.cross_entropy(masked_logits, masked_tokens)
        loss.backward()
        optimizer.step()

        return loss, masked_logits, masked_tokens

    def gp_train_step(self, optimizer, inputs, targets, mll):
        self.surrogate.zero_grad(set_to_none=True)
        self.surrogate.clear_cache()  # possibly unnecessary
        features = self.surrogate.get_features(
            inputs.to(self.surrogate.device), self.batch_size, transform=False
        )  # torch.Size([32, 16])

        dtype = features.dtype if (features.dtype is not torch.bool) else torch.float
        targets = self.surrogate.reshape_targets(targets).to(
            features.device, dtype=dtype
        )
        # if isinstance(surrogate, (SingleTaskGP, KroneckerMultiTaskGP)):
        # surrogate.set_train_data(features, targets, strict=False)

        output = self.surrogate.forward(features)
        loss = -mll(output, targets).mean()

        loss.backward()
        optimizer.step()
        return loss

    def fit(self):
        select_crit_key = "test_nll"

        X_train = self.dataset.train_dataset["samples"]
        Y_train = self.dataset.train_dataset["energies"]
        Y_train = self.surrogate.reshape_targets(Y_train)
        Y_train = Y_train.to(dtype=list(self.surrogate.parameters())[0].dtype)

        train_loader, test_loader = self.dataset.get_dataloader()

        print("\n---- preparing checkpoint ----")
        self.surrogate.eval()
        self.surrogate.requires_grad_(False)
        self.surrogate.set_train_data(X_train, Y_train, strict=False)
        start_metrics = {}
        start_metrics.update(self.surrogate.evaluate(test_loader))
        start_metrics["epoch"] = 0
        start_metrics.update(
            mlm_eval_epoch(
                self.language_model,
                test_loader,
                self.language_model.mask_ratio,
            )
        )

        best_score = start_metrics.get(select_crit_key, None)
        best_score_epoch = 0
        self.surrogate.cpu()  # avoid storing two copies of the weights on GPU
        best_weights = copy.deepcopy(self.surrogate.state_dict())
        self.surrogate.to(self.surrogate.device)
        if best_score is not None:
            print(f"starting val NLL: {best_score:.4f}")

        self.initialize_surrogate(X_train, Y_train)

        if hasattr(self.mll, "num_data"):
            self.mll.num_data = len(train_loader.dataset)

        stop_crit_key = "train_loss"
        best_loss, best_loss_epoch = None, 0
        stop = False

        gp_optimizer = torch.optim.Adam(self.surrogate.param_groups)
        gp_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gp_optimizer,
            patience=math.ceil(self.surrogate.patience / 2.0),
            threshold=1e-3,
        )
        # records = [start_metrics]

        print("\n---- fitting all params ----")
        for epoch_idx in range(self.surrogate.num_epochs):  # 128
            metrics = {}

            # train encoder through supervised MLL objective
            # if has_encoder and encoder_obj == 'mll':
            #     enc_sup_loss = fit_encoder_only(
            #         surrogate, gp_optimizer, mll, train_loader, num_epochs=1
            #     )
            # else:
            #     enc_sup_loss = 0.

            avg_train_loss = 0.0
            self.surrogate.train()
            for inputs, targets in train_loader:

                # train encoder through unsupervised MLM objective
                if self.encoder_obj == "mlm":
                    self.surrogate.encoder.requires_grad_(
                        True
                    )  # inputs = 32 x 36, mask_ratio=0.125, loss_scale=1.
                    mlm_loss, _, _ = self.mlm_train_step(
                        gp_optimizer,
                        inputs,
                        self.surrogate.encoder.mask_ratio,
                        loss_scale=1.0,
                    )
                # elif isinstance(surrogate.encoder, LanguageModel) and encoder_obj == 'lanmt':
                #     surrogate.encoder.requires_grad_(True)
                #     mlm_loss, _, _ = lanmt_train_step(
                #         surrogate.encoder.model, gp_optimizer, inputs, loss_scale=1.
                #     )
                # else:
                #     mlm_loss = torch.zeros(1, device=surrogate.device)

                # train all params through supervised MLL objective
                self.surrogate.requires_grad_(True)
                gp_loss = self.gp_train_step(gp_optimizer, inputs, targets, self.mll)

                avg_train_loss += (mlm_loss.detach() + gp_loss.detach()) / len(
                    train_loader
                )

            gp_lr_sched.step(avg_train_loss)

            metrics.update(
                {
                    "epoch": epoch_idx + 1,
                    "train_loss": avg_train_loss.item(),
                }
            )

            if (epoch_idx + 1) % self.surrogate.eval_period == 0:
                self.surrogate.requires_grad_(False)

                # update train features, use unaugmented train data for evaluation
                self.surrogate.eval()
                self.surrogate.set_train_data(X_train, Y_train, strict=False)
                metrics.update(self.surrogate.evaluate(test_loader))
                if self.encoder_obj == "mlm":
                    metrics.update(
                        mlm_eval_epoch(
                            self.language_model,
                            test_loader,
                            self.surrogate.encoder.mask_ratio,
                        )
                    )
                # elif has_encoder and encoder_obj == "lanmt":
                #     if val_loader is not None:
                #         metrics.update(
                #             lanmt_eval_epoch(
                #                 surrogate.encoder.model, val_loader, split="val"
                #             )
                #         )
                #     metrics.update(
                #         lanmt_eval_epoch(
                #             surrogate.encoder.model, test_loader, split="test"
                #         )
                #     )

            # use validation NLL for model selection
            select_crit = metrics.get(select_crit_key, None)
            if self.surrogate.early_stopping and select_crit is not None:
                # and select_crit is not None:
                assert (
                    self.surrogate.holdout_ratio > 0.0
                ), "Need validation data for early stopping"
                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=self.surrogate,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=select_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=self.surrogate.patience,
                    save_weights=True,
                )
            metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

            # use train loss to determine convergence
            stop_crit_key = "train_loss"
            stop_crit = metrics.get(stop_crit_key, None)
            if stop_crit is not None:
                best_loss, best_loss_epoch, _, stop = check_early_stopping(
                    model=self.surrogate,
                    best_score=best_loss,
                    best_epoch=best_loss_epoch,
                    best_weights=None,
                    curr_score=stop_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=self.surrogate.patience,
                    save_weights=False,
                )
            metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

            # records.append(metrics)
            log_prefix = "svgp"
            if len(log_prefix) > 0:
                metrics = {
                    "/".join((log_prefix, key)): val for key, val in metrics.items()
                }
            try:
                self.logger.log_metrics(metrics)
            except Exception:
                pass

            if stop:
                break

        if self.surrogate.early_stopping:
            print(f"\n---- loading checkpoint from epoch {best_score_epoch} ----")
            self.surrogate.load_state_dict(best_weights)
        self.surrogate.requires_grad_(False)
        self.surrogate.train()  # clear caches
        self.surrogate.clear_cache()
        self.surrogate.eval()
        self.surrogate.set_train_data(X_train, Y_train, strict=False)

        # return records
