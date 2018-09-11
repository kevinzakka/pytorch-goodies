"""
Useful definitions of common ml losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_loss(true, logits, pos_weight):
    """Computes the weighted binary cross-entropy loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        pos_weight: a scalar representing the weight attributed
            to the positive class. This is especially useful for
            an imbalanced dataset.

    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, true.float(), pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(true, logits, weights, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [2,]. The weights attributed
            to each class.
        ignore: the class index to ignore.

    Returns:
        ce_loss: the weighted binary cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        logits, true.squeeze(),
        ignore_index=ignore, weight=weights
    )
    return ce_loss


def dice_loss(true, logits, log=False, force_positive=False):
    """Computes the binary dice loss.

    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        log: whether to return the loss in log space.
        force_positive: whether to add 1 to the loss to prevent
            it from becoming negative.

    Returns:
        dice_loss: the binary dice loss.
    """
    eps = 1e-15
    dice_output = torch.sigmoid(logits)
    dice_target = (true == 1).float()
    intersection = (dice_output * dice_target).sum()
    union = dice_output.sum() + dice_target.sum() + eps
    dice_loss = 2 * intersection / union
    if force_positive:
        return (1 - dice_loss)
    if log:
        dice_loss = torch.log(dice_loss)
    return (-1 * dice_loss)


def jaccard_loss(true, pred):
    pass


def ce_dice(true, pred, log=False, w1=1, w2=1):
    pass


def ce_jaccard(true, pred, log=False, w1=1, w2=1):
    pass


def focal_loss(true, pred):
    pass
