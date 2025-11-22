# train_snake.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import logging
from tqdm import tqdm
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import (
    IMAGE_ROOT, 
    NUM_WORKERS, 
    EPOCHS_PHASE1, 
    EPOCHS_PHASE2, 
    LR_BACKBONE, 
    LR_HEADS,
    BATCH_SIZE,
    WEIGHT_DECAY
    )

from src.data_loader.data_loader import DataLoader                  
from src.model.model import TwoHeadConvNeXtV2


class Collate:
    def __init__(self, image_root: str):
        self.image_root = image_root

    def __call__(self, batch):
        images = []
        venomous_labels = []
        species_labels = []
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for sample in batch:
            img_path = sample.info.get("image_path")
            if img_path is None:
                raise ValueError("image_path nem található a Sample.info-ban!")
            
            full_path = os.path.join(self.image_root, img_path)
            img = DataLoader.load_image(full_path)
            images.append(img)

            venomous_labels.append(1.0 if sample.original_venomous else 0.0)
            species_labels.append(sample.original_class)

        return (
            torch.stack(images),
            torch.tensor(venomous_labels, dtype=torch.float32, device=DEVICE),
            torch.tensor(species_labels,   dtype=torch.long,   device=DEVICE),
        )

# Használat:
collate_fn = Collate(IMAGE_ROOT)

def create_torch_loader(samples, batch_size, shuffle):
    return TorchDataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,      # most már lehet 8 vagy több is!
        pin_memory=True,
        collate_fn=collate_fn,        # ← ez már pickle-elhető
    )

def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item() * 100

def species_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100



def train_model(data_loader: DataLoader, model: TwoHeadConvNeXtV2):

    train_samples = data_loader.get_training_set()
    val_samples   = data_loader.get_validation_set()

    train_loader = create_torch_loader(train_samples, BATCH_SIZE, shuffle=True)
    val_loader   = create_torch_loader(val_samples,   BATCH_SIZE, shuffle=False)

    # Lossok
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_sp  = nn.CrossEntropyLoss(label_smoothing=0.05)


    logging.info("1. FÁZIS: Csak a fejek edzése (backbone fagyasztva)")
    model.freeze_backbone()

    optimizer_heads = optim.AdamW(
        list(model.binary_head.parameters()) + list(model.species_head.parameters()),
        lr=LR_HEADS,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_heads = optim.lr_scheduler.ReduceLROnPlateau(optimizer_heads, patience=3, factor=0.5)

    for epoch in range(1, EPOCHS_PHASE1 + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase1 Epoch {epoch}"):
            bin_logit, sp_logit = model(imgs)

            loss_bin = criterion_bin(bin_logit, ven_lbl)
            loss_sp  = criterion_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp

            optimizer_heads.zero_grad()
            loss.backward()
            optimizer_heads.step()

            epoch_loss += loss.item()

        # Validáció
        val_loss, val_bin_acc, val_sp_acc = validate(model, val_loader, criterion_bin, criterion_sp)
        scheduler_heads.step(val_loss)

        logging.info(f"Phase1 | Epoch {epoch:02d} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
                     f"ValLoss {val_loss:.4f} | VenomousAcc {val_bin_acc:.2f}% | SpeciesAcc {val_sp_acc:.2f}%")

        torch.save(model.state_dict(), f"snake_phase1_epoch{epoch}.pth")


    logging.info("=== 2. FÁZIS: Teljes modell finetuning ===")
    model.unfreeze_backbone()

    optimizer_full = optim.AdamW([
        {"params": model.backbone.parameters(),     "lr": LR_BACKBONE},
        {"params": model.binary_head.parameters(),  "lr": LR_HEADS * 0.1},
        {"params": model.species_head.parameters(), "lr": LR_HEADS * 0.1},
    ], weight_decay=WEIGHT_DECAY)

    scheduler_full = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=EPOCHS_PHASE2)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase2 Epoch {epoch}"):
            bin_logit, sp_logit = model(imgs)

            loss_bin = criterion_bin(bin_logit, ven_lbl)
            loss_sp  = criterion_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp

            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()

            epoch_loss += loss.item()

        val_loss, val_bin_acc, val_sp_acc = validate(model, val_loader, criterion_bin, criterion_sp)
        scheduler_full.step()

        logging.info(f"Phase2 | Epoch {epoch:02d} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
                     f"ValLoss {val_loss:.4f} | VenomousAcc {val_bin_acc:.2f}% | SpeciesAcc {val_sp_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "snake_best_full.pth")
            logging.info("   Új legjobb modell mentve!")

    logging.info("Tanítás kész! Legjobb modell: snake_best_full.pth")


def validate(model, val_loader, crit_bin, crit_sp):
    model.eval()
    total_loss = 0.0
    bin_acc = sp_acc = 0.0
    with torch.no_grad():
        for imgs, ven_lbl, sp_lbl in val_loader:
            bin_logit, sp_logit = model(imgs)

            loss_bin = crit_bin(bin_logit, ven_lbl)
            loss_sp  = crit_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp
            total_loss += loss.item()

            bin_acc += binary_accuracy(bin_logit, ven_lbl)
            sp_acc  += species_accuracy(sp_logit, sp_lbl)

    return (total_loss / len(val_loader),
            bin_acc / len(val_loader),
            sp_acc  / len(val_loader))