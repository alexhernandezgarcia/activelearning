import math

import numpy as np
import torch
import torchvision
import wandb

from torch.nn import functional as F
from torch import LongTensor

# from lambo import transforms as gfp_transform/s, dataset as gfp_dataset
# from lambo.models.shared_elements import check_early_stopping
# from lambo.utils import str_to_tokens


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


def mlm_train_step(model, optimizer, token_batch, mask_ratio, loss_scale=1.0):
    optimizer.zero_grad(set_to_none=True)

    # replace random tokens with mask token
    mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)  # (32, 5)
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


def mlm_eval_epoch(model, eval_loader, mask_ratio, split=None):
    metrics = dict(
        perplexity=0.0,
    )
    model.eval()  # LanguageModel
    for minibatch in eval_loader:
        if isinstance(minibatch, tuple):
            token_batch = minibatch[0]  # torch.Size([32, 36])
        else:
            assert torch.is_tensor(minibatch)
            token_batch = minibatch
        # token_batch is padded states
        # replace random tokens with mask token
        mask_idxs = sample_mask(token_batch, model.tokenizer, mask_ratio)  # (32, 5)
        masked_token_batch = token_batch.clone().to(
            model.device
        )  # torch.Size([32, 36])
        np.put_along_axis(
            masked_token_batch, mask_idxs, model.tokenizer.masking_idx, axis=1
        )

        # get predicted logits for masked tokens
        logits, _ = model.logits_from_tokens(
            masked_token_batch
        )  # torch.Size([32, 36, 26]) HUGE
        vocab_size = logits.shape[-1]
        masked_logits = np.take_along_axis(logits, mask_idxs[..., None], axis=1).view(
            -1, vocab_size
        )  # torch.Size([160, 26])

        # use the ground-truth tokens as labels
        masked_tokens = np.take_along_axis(token_batch, mask_idxs, axis=1)
        masked_tokens = masked_tokens.view(-1).to(model.device)  # torch.Size([225])

        # logging
        log_prob = F.log_softmax(masked_logits, dim=-1)
        log_prob = np.take_along_axis(
            log_prob, masked_tokens.cpu().numpy()[..., None], axis=1
        )  # torch.Size([160, 1])
        metrics["perplexity"] += 2 ** (-(log_prob / math.log(2)).mean().detach()) / len(
            eval_loader
        )

    metrics = {key: val.item() for key, val in metrics.items()}
    metrics = {f"{split}_{key}": val for key, val in metrics.items()}

    return metrics
