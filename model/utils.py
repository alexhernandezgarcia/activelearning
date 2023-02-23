import torch
import torch.nn as nn
from .lm_elements import FunctionHead, LengthHead, PositionalEncoding, LengthTransform


@torch.jit.script
def fused_swish(x):
    return x * x.sigmoid()


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Apply(nn.Module):
    def __init__(self, module, dim=0):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return xs


class MaskLayerNorm1d(nn.LayerNorm):
    """
    Custom masked layer-norm layer
    """

    def forward(self, inp: tuple):
        x, mask = inp

        batch_size, num_channels, num_tokens = x.shape
        reshaped_mask = mask[:, None]

        sum_dims = list(range(len(x.shape)))[1:]
        xsum = x.sum(dim=sum_dims, keepdims=True)
        xxsum = (x * x).sum(dim=sum_dims, keepdims=True)
        numel_notnan = reshaped_mask.sum(dim=sum_dims, keepdims=True) * num_channels

        xmean = xsum / numel_notnan
        xxmean = xxsum / numel_notnan

        var = xxmean - (xmean * xmean)
        std = var.clamp(self.eps) ** 0.5
        ratio = self.weight / std

        output = (x - xmean) * ratio + self.bias

        return output, mask


class MaskBatchNormNd(nn.BatchNorm1d):
    """n-dimensional batchnorm that excludes points outside the mask from the statistics"""

    def forward(self, inp):
        """input (*, c), (*,) computes statistics averaging over * within the mask"""
        x, mask = inp
        sum_dims = list(range(len(x.shape)))[:-1]
        x_or_zero = torch.where(
            mask.unsqueeze(-1) > 0, x, torch.zeros_like(x)
        )  # remove nans
        if self.training or not self.track_running_stats:
            xsum = x_or_zero.sum(dim=sum_dims)
            xxsum = (x_or_zero * x_or_zero).sum(dim=sum_dims)
            numel_notnan = (mask).sum()
            xmean = xsum / numel_notnan
            sumvar = xxsum - xsum * xmean
            unbias_var = sumvar / (numel_notnan - 1)
            bias_var = sumvar / numel_notnan
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * xmean.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
        else:
            xmean, bias_var = self.running_mean, self.running_var
        std = bias_var.clamp(self.eps) ** 0.5
        ratio = self.weight / std
        output = x_or_zero * ratio + (self.bias - xmean * ratio)
        return (output, mask)


class mResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layernorm,
        dropout_p=0.1,
        act_fn="swish",
        stride=1,
    ):
        super().__init__()
        self.conv_1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            stride=stride,
            bias=False,
        )
        self.conv_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding="same", stride=stride
        )
        if layernorm:
            self.norm_1 = MaskLayerNorm1d(normalized_shape=[in_channels, 1])
            self.norm_2 = MaskLayerNorm1d(normalized_shape=[out_channels, 1])
        else:
            self.norm_1 = MaskBatchNormNd(in_channels)
            self.norm_2 = MaskBatchNormNd(out_channels)

        # TODO: choose which activation to use
        if act_fn == "swish":
            self.act_fn = fused_swish
        else:
            self.act_fn = nn.ReLU(inplace=True)

        if not in_channels == out_channels:
            self.proj = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same", stride=1
            )
        else:
            self.proj = None

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs):
        # assumes inputs are already properly masked
        resid, mask = inputs
        x, _ = self.norm_1((resid, mask))
        x = mask[:, None] * x
        x = self.act_fn(x)
        x = self.conv_1(x)
        x = mask[:, None] * x

        x, _ = self.norm_2((x, mask))
        x = mask[:, None] * x
        x = self.act_fn(x)
        x = self.conv_2(x)

        if self.proj is not None:
            resid = self.proj(resid)

        x = mask[:, None] * (x + resid)

        return self.dropout(x), mask


class mCNN(nn.Module):
    """1d CNN for sequences like CNN, but includes an additional mask
    argument (bs,n) that specifies which elements in the sequence are
    merely padded values."""

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    def __init__(
        self,
        tokenizer,
        max_len,
        embed_dim=128,
        kernel_size=5,
        p=0.1,
        out_dim=1,
        layernorm=False,
        latent_dim=8,
        max_len_delta=2,
        num_heads=2,
        **kwargs
    ):
        super().__init__()
        vocab_size = len(tokenizer.full_vocab)
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=tokenizer.padding_idx
        )
        self.pos_encoder = PositionalEncoding(embed_dim, p, max_len, batch_first=True)
        self.encoder = nn.Sequential(
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
            # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
            mResidualBlock(embed_dim, latent_dim, kernel_size, layernorm, p),
            # Apply(nn.Dropout(p=p)),
            Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        )
        # self.decoder = nn.Sequential(
        #     Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,N,C) -> (B,C,N)
        #     mResidualBlock(2 * latent_dim, embed_dim, kernel_size, layernorm, p),
        #     mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
        #     mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm, p),
        #     # mResidualBlock(embed_dim, embed_dim, kernel_size, layernorm),
        #     # Apply(nn.Dropout(p=p)),
        #     Apply(Expression(lambda x: x.permute(0, 2, 1))),  # (B,C,N) -> (B,N,C)
        # )

        self.length_transform = LengthTransform()
        self.function_head = FunctionHead(
            latent_dim, out_dim, kernel_size, layernorm, p, None, type="conv"
        )
        self.length_head = LengthHead(latent_dim, max_len_delta)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.tokenizer = tokenizer
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.max_len_delta = max_len_delta

    def enc_tok_features(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, : self.max_len + 1]

        src_tok_features = self.embedding(src_tok_idxs) * math.sqrt(self.embed_dim)
        src_tok_features = self.pos_encoder(src_tok_features)
        src_mask = (src_tok_idxs != self.tokenizer.padding_idx).float()
        src_tok_features, _ = self.encoder((src_tok_features, src_mask))
        return src_tok_features, src_mask

    # def dec_tok_features(self, src_tok_features, src_mask, lat_tok_features=None, tgt_lens=None):
    #     # internal features from function head
    #     if lat_tok_features is None:
    #         lat_tok_features, _ = self.function_head(
    #             src_tok_features, padding_mask=src_mask, pooling_mask=src_mask
    #         )

    #     len_delta_logits = self.length_head(src_tok_features, src_mask)
    #     # predict target seq length if unknown
    #     if tgt_lens is None:
    #         src_lens = src_mask.float().sum(-1)
    #         tgt_lens = self.length_head.sample(src_lens, len_delta_logits)

    #     tgt_tok_features, tgt_mask = self.length_transform(
    #         src_tok_features=torch.cat([src_tok_features, lat_tok_features], dim=-1),
    #         src_mask=src_mask,
    #         tgt_lens=tgt_lens
    #     )
    #     tgt_tok_features, _ = self.decoder((tgt_tok_features, tgt_mask))

    #     return lat_tok_features, tgt_tok_features, tgt_mask, len_delta_logits

    def tgt_tok_logits(self, tgt_tok_features):
        reshaped_features = tgt_tok_features.flatten(end_dim=-2)
        logits = self.lm_head(reshaped_features)
        logits = logits.view(*tgt_tok_features.shape[:-1], -1)
        return logits

    def forward(self, src_tok_idxs):
        if src_tok_idxs.size(1) > self.max_len:
            src_tok_idxs = src_tok_idxs[:, : self.max_len + 1]
        src_tok_features, src_mask = self.enc_tok_features(src_tok_idxs)
        pooling_mask = src_mask * src_tok_idxs.ne(self.tokenizer.eos_idx)
        _, pooled_features = self.function_head(
            src_tok_features, src_mask, pooling_mask
        )
        return pooled_features

    def param_groups(self, lr, weight_decay=0.0):
        shared_group = dict(
            params=[], lr=lr, weight_decay=weight_decay, betas=(0.0, 1e-2)
        )
        other_group = dict(params=[], lr=lr, weight_decay=weight_decay)

        shared_names = ["embedding", "pos_encoder", "encoder", "function_head"]
        for p_name, param in self.named_parameters():
            prefix = p_name.split(".")[0]
            if prefix in shared_names:
                shared_group["params"].append(param)
            else:
                other_group["params"].append(param)

        return shared_group, other_group
