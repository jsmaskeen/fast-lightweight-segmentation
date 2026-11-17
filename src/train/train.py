import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from src.utils.seed import seed_everything
from src.data.dataset import BUSIDataset, get_train_transforms
from src.train.models import get_model
from src.train.losses import BCEDiceLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train FP32 BUSI U-Net models")
    parser.add_argument("--model", type=str, required=True, choices=["r34", "mbv3s"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs (default from train.yaml)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default from train.yaml)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default from train.yaml)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience on val Dice")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to train config")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_val(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    lesion_dice_scores = []
    normal_fp_count_50px = 0
    normal_total = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        classes = batch["class_label"]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)

        preds = (logits > 0.0).float()
        b = images.size(0)
        for i in range(b):
            cls = classes[i]
            p = preds[i].view(-1)
            t = masks[i].view(-1)
            p_sum = p.sum().item()
            t_sum = t.sum().item()

            if cls in ["benign", "malignant"]:
                # Lesion scan
                intersection = (p * t).sum().item()
                if p_sum == 0 and t_sum == 0:
                    lesion_dice_scores.append(1.0)
                elif (p_sum == 0 and t_sum > 0) or (p_sum > 0 and t_sum == 0):
                    lesion_dice_scores.append(0.0)
                else:
                    lesion_dice_scores.append((2.0 * intersection) / (p_sum + t_sum))
            else:
                # Normal scan (ground truth has no lesion)
                normal_total += 1
                if p_sum > 50:
                    normal_fp_count_50px += 1

    val_loss = running_loss / len(dataloader.dataset)
    val_dice = sum(lesion_dice_scores) / len(lesion_dice_scores) if lesion_dice_scores else 0.0
    normal_fpr = normal_fp_count_50px / normal_total if normal_total > 0 else 0.0

    return val_loss, val_dice, normal_fpr


def train_model(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    seed_everything(seed)

    epochs = args.epochs or cfg.get("epochs", 100)
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    lr = args.lr or float(cfg.get("learning_rate", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    patience = args.patience or cfg.get("patience", 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training {args.model} on {device} (seed={seed}) ---")

    # Datasets and loaders
    train_dataset = BUSIDataset(split="train", transform=get_train_transforms())
    val_dataset = BUSIDataset(split="val", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=(device.type == "cuda")
    )

    print(f"Loaded {len(train_dataset)} train samples and {len(val_dataset)} val samples.")

    # Model, loss, optimizer, scheduler
    model = get_model(args.model, pretrained=True).to(device)
    criterion = BCEDiceLoss(
        bce_weight=cfg["loss"]["bce_weight"],
        dice_weight=cfg["loss"]["dice_weight"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(cfg.get("min_learning_rate", 1e-6))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path("models/fp32")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / f"{args.model}_best.pt"

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = results_dir / f"train_{args.model}.csv"

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "normal_fpr", "lr"])

    best_val_dice = -1.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, normal_fpr = evaluate_val(model, val_loader, criterion, device)
        curr_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Log to CSV
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_dice:.5f}", f"{normal_fpr:.4f}", f"{curr_lr:.7f}"])

        improved = val_dice > best_val_dice
        status = ""
        if improved:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "model_state_dict": model.state_dict(),
                "val_dice": val_dice,
                "val_loss": val_loss,
                "normal_fpr": normal_fpr,
                "seed": seed,
            }, best_ckpt_path)
            status = " [BEST SAVED]"
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | FPR@50px: {normal_fpr:.3f}{status}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement on val Dice.")
            break

    print(f"\n--- Training Complete for {args.model} ---")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path.resolve()}")

    assert best_val_dice >= 0.70, f"Gate 2 Failure: Val Dice {best_val_dice:.4f} is below required threshold 0.70"
    print(f"Gate 2 Check PASSED: Val Dice {best_val_dice:.4f} >= 0.70")
    return best_val_dice


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
