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
from src.distil.loss import KnowledgeDistillationLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Distil high-capacity teacher into lightweight student")
    parser.add_argument("--teacher-model", type=str, default="r34", choices=["r34", "mbv3s"], help="Teacher model architecture")
    parser.add_argument("--teacher-ckpt", type=str, default="models/fp32/r34_best.pt", help="Teacher checkpoint path")
    parser.add_argument("--student-model", type=str, default="mbv3s", choices=["r34", "mbv3s"], help="Student model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of distillation training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight balance between task loss and KD loss")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--out-dir", type=str, default="models/distil", help="Output directory for distilled model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Train config path")
    return parser.parse_args()


def train_one_epoch_distil(teacher, student, dataloader, criterion, optimizer, scaler, device):
    student.train()
    teacher.eval()
    running_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Student forward pass with autocast
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            student_logits = student(images)
            loss = criterion(student_logits, teacher_logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_student(student, dataloader, criterion, device):
    student.eval()
    running_loss = 0.0
    lesion_dice_scores = []
    normal_fp_count_50px = 0
    normal_total = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        classes = batch["class_label"]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = student(images)
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
                intersection = (p * t).sum().item()
                if p_sum == 0 and t_sum == 0:
                    lesion_dice_scores.append(1.0)
                elif (p_sum == 0 and t_sum > 0) or (p_sum > 0 and t_sum == 0):
                    lesion_dice_scores.append(0.0)
                else:
                    lesion_dice_scores.append((2.0 * intersection) / (p_sum + t_sum))
            else:
                normal_total += 1
                if p_sum > 50:
                    normal_fp_count_50px += 1

    val_loss = running_loss / len(dataloader.dataset)
    val_dice = sum(lesion_dice_scores) / len(lesion_dice_scores) if lesion_dice_scores else 0.0
    normal_fpr = normal_fp_count_50px / normal_total if normal_total > 0 else 0.0

    return val_loss, val_dice, normal_fpr


def distil_model(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Distillation running on device: {device}")

    epochs = args.epochs if args.epochs is not None else cfg.get("epochs", 10)
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size", 16)
    lr = args.lr if args.lr is not None else cfg.get("learning_rate", 3e-4)
    patience = args.patience if args.patience is not None else cfg.get("patience", 10)
    num_workers = cfg.get("num_workers", 2)

    # 1. Datasets and Loaders
    train_transform = get_train_transforms(image_size=256)
    train_dataset = BUSIDataset(split="train", transform=train_transform)
    val_dataset = BUSIDataset(split="val", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # 2. Setup Teacher
    print(f"\n--- Initializing Teacher: {args.teacher_model} ---")
    teacher = get_model(args.teacher_model, pretrained=False).to(device)
    teacher_ckpt = Path(args.teacher_ckpt)
    if teacher_ckpt.exists():
        print(f"Loading teacher weights from {teacher_ckpt}")
        ckpt_data = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
        teacher.load_state_dict(state_dict)
    else:
        print(f"Warning: Teacher checkpoint {teacher_ckpt} not found. Using randomly initialized teacher.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Setup Student
    print(f"\n--- Initializing Student: {args.student_model} ---")
    student = get_model(args.student_model, pretrained=True).to(device)

    # 4. Losses & Optimizer
    kd_criterion = KnowledgeDistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        bce_weight=cfg.get("loss", {}).get("bce_weight", 0.5),
        dice_weight=cfg.get("loss", {}).get("dice_weight", 0.5),
    )
    val_criterion = BCEDiceLoss(
        bce_weight=cfg.get("loss", {}).get("bce_weight", 0.5),
        dice_weight=cfg.get("loss", {}).get("dice_weight", 0.5),
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.get("min_learning_rate", 1e-6)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / f"{args.student_model}_distilled_best.pt"
    log_csv_path = out_dir / f"{args.student_model}_distilled_train.csv"

    best_val_dice = -1.0
    epochs_no_improve = 0

    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_normal_fpr", "lr"])

    print(f"\nStarting distillation training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_distil(teacher, student, train_loader, kd_criterion, optimizer, scaler, device)
        val_loss, val_dice, val_normal_fpr = evaluate_student(student, val_loader, val_criterion, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Normal FPR: {val_normal_fpr:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        with open(log_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_dice, val_normal_fpr, current_lr])

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_dice,
                    "val_loss": val_loss,
                    "val_normal_fpr": val_normal_fpr,
                    "teacher_model": args.teacher_model,
                    "student_model": args.student_model,
                    "temperature": args.temperature,
                    "alpha": args.alpha,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch} (No improvement for {patience} epochs).")
                break

    print(f"\nDistillation complete. Best checkpoint saved at: {best_ckpt_path} (Val Dice: {best_val_dice:.4f})")
    return best_ckpt_path


def main():
    args = parse_args()
    distil_model(args)


if __name__ == "__main__":
    main()
