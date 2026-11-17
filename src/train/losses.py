import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Soft Dice loss:
    Loss = 0.5 * BCEWithLogits + 0.5 * SoftDiceLoss
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_fn(logits, targets)

        probs = torch.sigmoid(logits)
        b = logits.size(0)
        probs_flat = probs.view(b, -1)
        targets_flat = targets.view(b, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        dice_loss = (1.0 - dice).mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


def compute_batch_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0) -> float:
    """
    Compute mean per-image Dice coefficient across a batch.
    Thresholding at logits > 0.0 is equivalent to probability > 0.5.
    """
    preds = (logits > threshold).float()
    b = logits.size(0)
    dice_scores = []

    for i in range(b):
        p = preds[i].view(-1)
        t = targets[i].view(-1)

        p_sum = p.sum().item()
        t_sum = t.sum().item()
        intersection = (p * t).sum().item()

        if p_sum == 0 and t_sum == 0:
            dice_scores.append(1.0)
        elif (p_sum == 0 and t_sum > 0) or (p_sum > 0 and t_sum == 0):
            dice_scores.append(0.0)
        else:
            dice_scores.append((2.0 * intersection) / (p_sum + t_sum))

    return sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
