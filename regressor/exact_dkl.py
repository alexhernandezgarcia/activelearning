from .dkl import DeepKernelRegressor
from gpytorch.mlls import ExactMarginalLogLikelihood


class ExactDKL(DeepKernelRegressor):
    def __init__(
        self,
        logger,
        device,
        dataset,
        config_model,
        config_env,
        surrogate,
        float_precision,
        tokenizer,
        checkpoint,
        encoder_obj="mlm",
        optim="adam",
        lr_sched_type="plateau",
        batch_size=32,
        **kwargs
    ):
        super().__init__(
            logger,
            device,
            dataset,
            config_model,
            config_env,
            surrogate,
            float_precision,
            tokenizer,
            checkpoint,
            encoder_obj,
            optim,
            lr_sched_type,
            batch_size,
            **kwargs
        )

    def initialize_surrogate(self, X_train, Y_train):
        self.mll = ExactMarginalLogLikelihood(self.surrogate.likelihood, self.surrogate)
        self.mll.to(self.surrogate.device)
