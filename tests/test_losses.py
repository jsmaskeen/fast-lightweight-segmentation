import torch
from src.train.losses import BCEDiceLoss, compute_batch_dice


def test_bce_dice_loss_shape_and_grad():
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(4, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_bce_dice_loss_perfect_prediction():
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    targets = torch.ones(2, 1, 32, 32)
    # Logits strongly positive where target is 1
    logits = 10.0 * torch.ones(2, 1, 32, 32)

    loss = loss_fn(logits, targets)
    # BCE should be near 0, Dice loss should be near 0
    assert loss.item() < 0.01


def test_compute_batch_dice():
    logits = torch.tensor([[[[10.0]]], [[[-10.0]]]])
    targets = torch.tensor([[[[1.0]]], [[[-1.0]]]])  # target for idx 1 is empty (<=0)
    # idx 0: pred=1, tgt=1 -> dice=1.0
    # idx 1: pred=0, tgt=0 -> dice=1.0
    score = compute_batch_dice(logits, targets > 0)
    assert score == 1.0
