import torch
import gpytorch
import hydra
from gflownet.utils.common import set_device, set_float_precision
from ..utils.dkl_utils import batched_call
from gpytorch.settings import cholesky_jitter
from gpytorch.lazy import ConstantDiagLazyTensor
import math
from torch import LongTensor
from gpytorch import lazify
import numpy as np
from torch.nn import functional as F


class Tokenizer:
    def __init__(self, non_special_vocab):
        self.non_special_vocab = non_special_vocab
        # skipped UNK and 0
        self.special_vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        self.full_vocab = self.special_vocab + self.non_special_vocab
        self.lookup = {a: i for (i, a) in enumerate(self.full_vocab)}
        self.inverse_lookup = {i: a for (i, a) in enumerate(self.full_vocab)}
        padding_token = "[PAD]"
        masking_token = "[MASK]"
        bos_token = "[CLS]"
        eos_token = "[SEP]"
        self.padding_idx = self.lookup[padding_token]
        self.masking_idx = self.lookup[masking_token]
        self.bos_idx = self.lookup[bos_token]
        self.eos_idx = self.lookup[eos_token]
        self.sampling_vocab = non_special_vocab


class DeepKernelRegressor:
    def __init__(
        self,
        logger,
        device,
        dataset,
        encoder,
        surrogate,
        env_vocab,
        float_precision,
        encoder_obj="mlm",
        batch_size=32,
    ):
        self.logger = logger
        self.progress = self.logger.progress

        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

        # Dataset
        self.dataset = dataset
        self.n_samples = dataset.n_samples

        tokenizer = Tokenizer(env_vocab)

        self.encoder_config = encoder
        self.encoder = hydra.utils.instantiate(encoder, tokenizer=tokenizer)
        self.encoder_obj = encoder_obj

        self.surrogate_config = surrogate
        self.surrogate = hydra.utils.instantiate(
            surrogate, tokenizer=self.encoder.tokenizer, encoder=self.encoder
        )

        self.batch_size = batch_size

    def initialize_model(self, X_train, Y_train):
        train_loader, test_loader = self.dataset.get_dataloader()
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
            try:
                self.surrogate.train()
                self.surrogate.init_inducing_points(
                    init_features
                )  # botorch.SingleTaskVariationalGP based function
                self.initialize_var_dist_sgpr(
                    self.surrogate,
                    init_features,
                    Y_train.to(init_features),
                    noise_lb=1.0,
                )
                print("variational initialization successful")
            except Exception as exp:
                # logging.exception(exp)
                print("variational initialization failed")

    def initialize_var_dist_sgpr(self, model, train_x, train_y, noise_lb):
        """
        This is only intended for whitened variational distributions and gaussian likelihoods
        at present.

        \bar m = L^{-1} m
        \bar S = L^{-1} S L^{-T}

        where $LL^T = K_{uu}$.

        Thus, the optimal \bar m, \bar S are given by
        \bar S = L^T (K_{uu} + \sigma^{-2} K_{uv} K_{vu})^{-1} L
        \bar m = \bar S L^{-1} (K_{uv} y \sigma^{-2})
        """

        if isinstance(
            model.model.variational_strategy, IndependentMultitaskVariationalStrategy
        ):  # True
            ind_pts = (
                model.model.variational_strategy.base_variational_strategy.inducing_points
            )  # Parameter containing: torch.Size([64, 16])
            train_y = train_y.transpose(-1, -2).unsqueeze(-1)
            is_batch_model = True
        else:
            ind_pts = model.model.variational_strategy.inducing_points
            is_batch_model = False

        with cholesky_jitter(1e-4):
            kuu = model.model.covar_module(
                ind_pts
            ).double()  # <gpytorch.lazy.lazy_evaluated_kernel_tensor.LazyEvaluatedKernelTensor object at 0x7fa0aecc3c40>
            kuu_chol = (
                kuu.cholesky()
            )  # <gpytorch.lazy.triangular_lazy_tensor.TriangularLazyTensor object at 0x7fa0cc0de130>
            kuv = model.model.covar_module(ind_pts, train_x).double()

            # noise = model.likelihood.noise if not is_batch_model else model.likelihood.task_noises.unsqueeze(-1).unsqueeze(-1)

            if hasattr(model.likelihood, "noise"):  # False
                noise = model.likelihood.noise
            elif hasattr(model.likelihood, "task_noises"):
                noise = model.likelihood.task_noises.view(-1, 1, 1)
            else:
                raise AttributeError
            noise = noise.clamp(min=noise_lb).double()  # torch.Size([3, 1, 1])

            if len(train_y.shape) < len(kuv.shape):
                train_y = train_y.unsqueeze(-1)
            if len(noise.shape) < len(kuv.shape):
                noise = noise.unsqueeze(-1)

            data_term = kuv.matmul(train_y.double()) / noise  # torch.Size([3, 64, 1])
            # mean_term = kuu_chol.inv_matmul(data_term)
            if is_batch_model:  # True
                # TODO: clean this up a bit more
                noise_as_lt = ConstantDiagLazyTensor(
                    noise.squeeze(-1), diag_shape=kuv.shape[-1]
                )
                inner_prod = kuv.matmul(noise_as_lt).matmul(kuv.transpose(-1, -2))
                inner_term = inner_prod + kuu
            else:
                inner_term = kuv @ kuv.transpose(-1, -2) / noise + kuu

            s_mat = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(kuu_chol.evaluate())
            )  # torch.Size([3, 64, 64])
            s_root = lazify(s_mat).cholesky().evaluate()  # torch.Size([3, 64, 64])
            # mean_param = s_mat.matmul(mean_term)
            # the expression below is less efficient but probably more stable
            mean_param = kuu_chol.transpose(-1, -2).matmul(
                inner_term.inv_matmul(data_term)
            )

        mean_param = mean_param.to(train_y)  # torch.Size([3, 64, 1])
        s_root = s_root.to(train_y)  # torch.Size([3, 64, 64])

        # if not is_batch_model:
        #     model.model.variational_strategy._variational_distribution.variational_mean.data = mean_param.data.detach().squeeze()
        #     model.model.variational_strategy._variational_distribution.chol_variational_covar.data = s_root.data.detach()
        #     model.model.variational_strategy.variational_params_initialized.fill_(1)
        # else: #True
        model.model.variational_strategy.base_variational_strategy._variational_distribution.variational_mean.data = (
            mean_param.data.detach().squeeze()
        )
        model.model.variational_strategy.base_variational_strategy._variational_distribution.chol_variational_covar.data = (
            s_root.data.detach()
        )
        model.model.variational_strategy.base_variational_strategy.variational_params_initialized.fill_(
            1
        )

    def sample_mask(
        token_batch: LongTensor, tokenizer, mask_ratio: float = 0.125, mask_size=None
    ):
        """Chooses the lements ot be masked but ensures that none of the special tokens are masked.
        Args:
                token_batch: (batch_size, num_tokens)
                tokenizer: only necessary to avoid masking special tokens
                mask_ratio: proportion of tokens to mask
                mask_size: (optional) override mask_ratio with a specific mask size
        Returns:
                mask_idxs: (batch_size, mask_size) np.ndarray of position indexes to mask
        """
        if mask_size is None:  # mask_Size = 5
            mask_size = math.ceil(token_batch.shape[-1] * mask_ratio)
        # special_vocab = {'[UNK]', '0', '[MASK]', '[PAD]', '[SEP]', '[CLS]'}
        # special_idxs = [3, 25, 4, 0, 2, 1]
        # lookup: {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3, '[MASK]': 4, 'A': 5, 'R': 6, 'N': 7, 'D': 8, 'C': 9, 'E': 10, 'Q': 11, 'G': 12, 'H': 13, ...}
        special_idxs = torch.tensor(tokenizer.special_idxs).view(
            -1, 1, 1
        )  # torch.Size([6, 1, 1])
        is_non_special = (
            token_batch.ne(special_idxs).prod(dim=0).float()
        )  # torch.Size([32, 36])
        mask_weights = is_non_special / is_non_special.sum(
            dim=-1, keepdims=True
        )  # torch.Size([32, 36])
        mask_idxs = torch.multinomial(
            mask_weights, mask_size, replacement=False
        )  # torch.Size([32, 5])
        return mask_idxs.numpy()

    def mlm_train_step(self, model, optimizer, token_batch, mask_ratio, loss_scale=1.0):
        optimizer.zero_grad(set_to_none=True)

        # replace random tokens with mask token
        mask_idxs = self.sample_mask(
            token_batch, model.tokenizer, mask_ratio
        )  # (32, 5)
        masked_token_batch = token_batch.clone().to(model.device)
        np.put_along_axis(
            masked_token_batch, mask_idxs, model.tokenizer.masking_idx, axis=1
        )  # masking_idx = 4

        # get predicted logits for masked tokens
        logits, _ = model.logits_from_tokens(masked_token_batch)
        vocab_size = logits.shape[-1]
        masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(
            -1, vocab_size
        )  # torch.Size([160, 26])

        # use the ground-truth tokens as labels
        masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
        masked_tokens = masked_tokens.view(-1).to(model.device)  # torch.Size([160])

        loss = loss_scale * F.cross_entropy(masked_logits, masked_tokens)
        loss.backward()
        optimizer.step()

        return loss, masked_logits, masked_tokens

    def gp_train_step(self, optimizer, inputs, targets, mll):
        self.surrogate.zero_grad(set_to_none=True)
        self.surrogate.clear_cache()  # possibly unnecessary
        features = self.surrogate.get_features(
            inputs.to(self.surrogate.device), surrogate.bs, transform=False
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
        self.initialize_model()
        mll = VariationalELBO(self.likelihood, self.model, num_data=X_train.shape[0])
        mll.to(surrogate.device)
        if hasattr(mll, "num_data"):
            mll.num_data = len(train_loader.dataset)

        self.gp_optimizer = torch.optim.Adam(surrogate.param_groups)
        gp_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gp_optimizer, patience=math.ceil(surrogate.patience / 2.0), threshold=1e-3
        )

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
                        self.surrogate.encoder,
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
                gp_loss = self.gp_train_step(
                    self.surrogate, gp_optimizer, inputs, targets, mll
                )

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
                if val_loader is not None:
                    metrics.update(self.surrogate.evaluate(val_loader, split="val"))
                metrics.update(self.surrogate.evaluate(test_loader, split="test"))
                if encoder_obj == "mlm":
                    metrics.update(
                        mlm_eval_epoch(
                            surrogate.encoder,
                            test_loader,
                            surrogate.encoder.mask_ratio,
                            split="test",
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
            if surrogate.early_stopping and select_crit is not None:
                assert (
                    surrogate.holdout_ratio > 0.0
                ), "Need validation data for early stopping"
                best_score, best_score_epoch, best_weights, _ = check_early_stopping(
                    model=surrogate,
                    best_score=best_score,
                    best_epoch=best_score_epoch,
                    best_weights=best_weights,
                    curr_score=select_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=surrogate.patience,
                    save_weights=True,
                )
            metrics.update(dict(best_score=best_score, best_epoch=best_score_epoch))

            # use train loss to determine convergence
            stop_crit = metrics.get(stop_crit_key, None)
            if stop_crit is not None:
                best_loss, best_loss_epoch, _, stop = check_early_stopping(
                    model=surrogate,
                    best_score=best_loss,
                    best_epoch=best_loss_epoch,
                    best_weights=None,
                    curr_score=stop_crit,
                    curr_epoch=epoch_idx + 1,
                    patience=surrogate.patience,
                    save_weights=False,
                )
            metrics.update(dict(best_loss=best_loss, best_loss_epoch=best_loss_epoch))

            records.append(metrics)
            if len(log_prefix) > 0:
                metrics = {
                    "/".join((log_prefix, key)): val for key, val in metrics.items()
                }
            try:
                wandb.log(metrics)
            except Exception:
                pass

            if stop:
                break

        if surrogate.early_stopping:
            print(f"\n---- loading checkpoint from epoch {best_score_epoch} ----")
            surrogate.load_state_dict(best_weights)
        surrogate.requires_grad_(False)
        surrogate.train()  # clear caches
        surrogate.clear_cache()
        surrogate.eval()
        surrogate.set_train_data(X_train, Y_train, strict=False)

        # return records
