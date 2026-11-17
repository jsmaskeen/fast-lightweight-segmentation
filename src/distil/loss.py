import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for segmentation models.
    Combines:
      1. Hard ground-truth loss (BCE + Dice)
      2. Soft distillation loss between student logits and teacher logits (temperature-scaled KL / BCE)
      3. Optional feature hint / MSE loss between intermediate representations
    """
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)  # Weight for ground truth loss; (1 - alpha) for teacher KD loss
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce_fn = nn.BCEWithLogitsLoss()

    def task_loss(self, student_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined BCE + Dice loss on ground truth targets."""
        bce = self.bce_fn(student_logits, targets)

        probs = torch.sigmoid(student_logits)
        b = student_logits.size(0)
        probs_flat = probs.view(b, -1)
        targets_flat = targets.view(b, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth)
        dice_loss = (1.0 - dice).mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss

    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Distillation loss between soft teacher probability and soft student probability.
        Uses temperature-scaled binary cross-entropy with soft targets:
        loss = (T^2) * BCEWithLogits(student_logits / T, sigmoid(teacher_logits / T))
        """
        s_scaled = student_logits / self.temperature
        t_scaled = teacher_logits / self.temperature
        t_soft = torch.sigmoid(t_scaled)

        # Binary cross entropy with soft targets
        kd_loss = F.binary_cross_entropy_with_logits(s_scaled, t_soft)
        return (self.temperature ** 2) * kd_loss

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        hard_loss = self.task_loss(student_logits, targets)
        kd_loss = self.distillation_loss(student_logits, teacher_logits)
        return self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss
