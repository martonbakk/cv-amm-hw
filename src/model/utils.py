# train_snake.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import logging
from tqdm import tqdm
import os, sys
import random
import numpy as np
from src.data_loader.data_loader import ListDataset, Sample, basic_transform
from src.data_loader.augmentation import SnakeAugmentor
from PIL import Image

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

# Define device globally for consistency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

class Collate:
    """Collate function that handles on-demand augmentation without caching"""
    def __init__(self, image_root: str):
        self.image_root = image_root

    def __call__(self, batch):
        images = []
        venomous_labels = []
        species_labels = []

        for sample in batch:
            if sample.info.get('augmented', False):
                # Get source image path directly from metadata
                source_img_path = sample.info['aug_params']['source_image_path']
                full_path = os.path.join(self.image_root, source_img_path)
                
                # Load source image ON DEMAND
                img = Image.open(full_path).convert("RGB")
                img_np = np.array(img)
                
                # Apply augmentation
                try:
                    augmented_img = SnakeAugmentor.randaugment(
                        image=img_np,
                        n_transforms=sample.info['aug_params']['n_transforms'],
                        magnitude=sample.info['aug_params']['magnitude']
                    )
                    img = Image.fromarray(augmented_img)
                except Exception as e:
                    logging.info(f"Augmentation failed: {str(e)}. Using original.")
                    print("Augmentation failed, using original image.")
                finally:
                    img = basic_transform(img)
            else:
                # Regular sample - load normally
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
            torch.tensor(venomous_labels, dtype=torch.float32),
            torch.tensor(species_labels, dtype=torch.long),
        )

def create_torch_loader(samples, batch_size, shuffle):
    return TorchDataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,  # Enable pinned memory for faster GPU transfer
        collate_fn=Collate(IMAGE_ROOT),
    )

def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item() * 100

def species_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item() * 100

def train_model(data_loader: DataLoader, model: TwoHeadConvNeXtV2, augmentor=None):
    # Move model to GPU immediately
    model = model.to(DEVICE)
    
    train_samples = data_loader.get_training_set()
    val_samples = data_loader.get_validation_set()

    if augmentor is not None:
        augmented_samples = []
        for _ in range(augmentor.num_augmentations):
            # Randomly select a source sample
            source_sample = random.choice(train_samples)
            
            # Sample augmentation parameters
            rng = np.random.default_rng()
            n_t = max(0, min(round(augmentor.center_n_transforms + rng.normal(scale=augmentor.center_n_transforms*0.5)), 10))
            mag = max(0, min(round(augmentor.center_magnitude + rng.normal(scale=augmentor.center_magnitude*0.5)), 30))
            
            # Create augmented sample metadata
            aug_info = {
                **source_sample.info,
                'augmented': True,
                'aug_params': {
                    'n_transforms': n_t,
                    'magnitude': mag,
                    # CRITICAL: Store source image path directly
                    'source_image_path': source_sample.info['image_path']
                }
            }
            
            augmented_samples.append(
                Sample(
                    original_class=source_sample.original_class,
                    original_venomous=source_sample.original_venomous,
                    predicted_class=None,
                    predicted_venomous=None,
                    image=None,
                    info=aug_info
                )
            )
        
        # Combine and shuffle
        combined_samples = list(train_samples) + augmented_samples
        random.shuffle(combined_samples)
        train_samples = ListDataset(combined_samples)
        logging.info(f"Created {len(augmented_samples)} virtual augmented samples")

    train_loader = create_torch_loader(train_samples, BATCH_SIZE, shuffle=True)
    val_loader = create_torch_loader(val_samples, BATCH_SIZE, shuffle=False)

    # Loss functions
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_sp = nn.CrossEntropyLoss(label_smoothing=0.05)

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
            # Move batch to GPU
            imgs = imgs.to(DEVICE, non_blocking=True)
            ven_lbl = ven_lbl.to(DEVICE, non_blocking=True)
            sp_lbl = sp_lbl.to(DEVICE, non_blocking=True)

            bin_logit, sp_logit = model(imgs)
            loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
            loss_sp = criterion_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp

            optimizer_heads.zero_grad()
            loss.backward()
            optimizer_heads.step()

            epoch_loss += loss.item()

        # Validation
        val_loss, val_bin_acc, val_sp_acc = validate(model, val_loader, criterion_bin, criterion_sp)
        scheduler_heads.step(val_loss)

        logging.info(f"Phase1 | Epoch {epoch:02d} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
                     f"ValLoss {val_loss:.4f} | VenomousAcc {val_bin_acc:.2f}% | SpeciesAcc {val_sp_acc:.2f}%")

        torch.save(model.state_dict(), f"snake_phase1_epoch{epoch}.pth")

    logging.info("=== 2. FÁZIS: Teljes modell finetuning ===")
    model.unfreeze_backbone()

    optimizer_full = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.binary_head.parameters(), "lr": LR_HEADS * 0.1},
        {"params": model.species_head.parameters(), "lr": LR_HEADS * 0.1},
    ], weight_decay=WEIGHT_DECAY)

    scheduler_full = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=EPOCHS_PHASE2)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS_PHASE2 + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase2 Epoch {epoch}"):
            # Move batch to GPU
            imgs = imgs.to(DEVICE, non_blocking=True)
            ven_lbl = ven_lbl.to(DEVICE, non_blocking=True)
            sp_lbl = sp_lbl.to(DEVICE, non_blocking=True)

            bin_logit, sp_logit = model(imgs)
            loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
            loss_sp = criterion_sp(sp_logit, sp_lbl)
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

@torch.no_grad()
def validate(model, val_loader, crit_bin, crit_sp):
    model.eval()
    total_loss = 0.0
    bin_acc = sp_acc = 0.0
    
    for imgs, ven_lbl, sp_lbl in val_loader:
        # Move batch to GPU
        imgs = imgs.to(DEVICE, non_blocking=True)
        ven_lbl = ven_lbl.to(DEVICE, non_blocking=True)
        sp_lbl = sp_lbl.to(DEVICE, non_blocking=True)

        bin_logit, sp_logit = model(imgs)
        loss_bin = crit_bin(bin_logit.squeeze(), ven_lbl)
        loss_sp = crit_sp(sp_logit, sp_lbl)
        loss = loss_bin + loss_sp
        total_loss += loss.item()

        bin_acc += binary_accuracy(bin_logit, ven_lbl)
        sp_acc += species_accuracy(sp_logit, sp_lbl)

    return (total_loss / len(val_loader),
            bin_acc / len(val_loader),
            sp_acc / len(val_loader))