# train_snake.py
from typing import Optional
import torch
import torch.nn as nn
from typing import Optional
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import logging
from tqdm import tqdm
import os, sys
import random
import numpy as np
from src.data_loader.data_loader import ListDataset, Sample, basic_transform
from src.data_loader.augmentation import SnakeAugmentor, Augmentor
from PIL import Image
import optuna
from optuna.trial import Trial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import (
    IMAGE_ROOT, 
    NUM_WORKERS, 
    EPOCHS_PHASE1, 
    EPOCHS_PHASE2, 
    LR_BACKBONE, 
    LR_HEADS,
    BATCH_SIZE,
    WEIGHT_DECAY,
    LABEL_SMOOTHING
    )

from src.data_loader.data_loader import DataLoader                  
from src.model.model import TwoHeadConvNeXtV2


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
                
                # Apply augmentation
                try:
                    augmented_img = SnakeAugmentor.randaugment(
                        image=img,
                        n_transforms=sample.info.get('aug_params', {}).get('n_transforms', 2),
                        magnitude=sample.info.get('aug_params', {}).get('magnitude', 10)

                    )
                except Exception as e:
                    logging.info(f"Augmentation failed: {str(e)}. Using original.")
                    print("Augmentation failed, using original image.")
                finally:
                    img = basic_transform(augmented_img)
            else:
                # Regular sample - load normally
                img_path = sample.info.get("image_path", None)

                if img_path is None:
                    raise ValueError("image_path nem található a Sample.info-ban!")
                full_path = os.path.join(self.image_root, img_path)
                img = DataLoader.load_image(full_path)

            
            images.append(img)
            venomous_labels.append(sample.original_venomous)
            species_labels.append(sample.original_class)

        return (
            torch.stack(images),
            torch.tensor(venomous_labels, dtype=torch.float32),
            torch.tensor(species_labels, dtype=torch.long),
        )

def create_torch_loader(samples, batch_size, shuffle) -> TorchDataLoader:
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

def train_model(
    data_loader: DataLoader,
    model: TwoHeadConvNeXtV2,
    augmentor: Optional[Augmentor] = None,
    lr_heads: float = LR_HEADS,
    lr_backbone: float = LR_BACKBONE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    epochs_phase1: int =EPOCHS_PHASE1,
    epochs_phase2: int =EPOCHS_PHASE2,
    save_models: bool =True,
    trial: Optional[Trial] = None
):
    
    train_samples: ListDataset = data_loader.get_training_set()
    val_samples: ListDataset = data_loader.get_validation_set()

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
                    info=aug_info
                )
            )
        
        # Combine and shuffle
        combined_samples = list(train_samples) + augmented_samples
        random.shuffle(combined_samples)
        train_samples = ListDataset(combined_samples)
        logging.info(f"Created {len(augmented_samples)} virtual augmented samples")

    # Subsample if optuna is running the function
    if trial is not None:
        train_samples = ListDataset(train_samples[:10000])
        val_samples = ListDataset(val_samples[:2000])
        logging.info("Optuna tuning: Using reduced dataset for faster training")

    train_loader: TorchDataLoader = create_torch_loader(train_samples, BATCH_SIZE, shuffle=True)
    val_loader: TorchDataLoader = create_torch_loader(val_samples, BATCH_SIZE, shuffle=False)

    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_sp = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    logging.info("1. FÁZIS: Csak a fejek edzése (backbone fagyasztva)")
    model.freeze_backbone()

    optimizer_heads = optim.AdamW(
        list(model.binary_head.parameters()) + list(model.species_head.parameters()),
        lr=lr_heads,
        weight_decay=weight_decay,
    )
    scheduler_heads = optim.lr_scheduler.ReduceLROnPlateau(optimizer_heads, patience=3, factor=0.5)

    # define gradient scaler for mixed precision
    scaler = torch.GradScaler()

    for epoch in range(1, epochs_phase1 + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase1 Epoch {epoch}"):
            # Move batch to GPU
            imgs = imgs.to(model.device, non_blocking=True)
            ven_lbl = ven_lbl.to(model.device, non_blocking=True)
            sp_lbl = sp_lbl.to(model.device, non_blocking=True)

            optimizer_heads.zero_grad()
            with torch.autocast(device_type=model.device.type):
                bin_logit, sp_logit = model(imgs)
                loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)
                loss = loss_bin + loss_sp
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_heads)
            scaler.update()

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
        {"params": model.backbone.parameters(), "lr": lr_backbone},
        {"params": model.binary_head.parameters(), "lr": lr_heads * 0.1},
        {"params": model.species_head.parameters(), "lr": lr_heads * 0.1},
    ], weight_decay=weight_decay)

    scheduler_full = optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=epochs_phase2)

    best_val_loss = float("inf")
    for epoch in range(1, epochs_phase2 + 1):
        model.train()
        epoch_loss = 0.0
        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase2 Epoch {epoch}"):
            # Move batch to GPU
            imgs = imgs.to(model.device, non_blocking=True)
            ven_lbl = ven_lbl.to(model.device, non_blocking=True)
            sp_lbl = sp_lbl.to(model.device, non_blocking=True)

            optimizer_full.zero_grad()
            with torch.autocast(device_type=model.device.type):
                bin_logit, sp_logit = model(imgs)
                loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)
                loss = loss_bin + loss_sp
            scaler.scale(loss).backward()
            scaler.step(optimizer_full)
            scaler.update()

            epoch_loss += loss.item()

        val_loss, val_bin_acc, val_sp_acc = validate(model, val_loader, criterion_bin, criterion_sp)
        scheduler_full.step()

        logging.info(f"Phase2 | Epoch {epoch:02d} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
                     f"ValLoss {val_loss:.4f} | VenomousAcc {val_bin_acc:.2f}% | SpeciesAcc {val_sp_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_models:
                torch.save(model.state_dict(), "snake_best_full.pth")
                logging.info("   Új legjobb modell mentve!")
        if trial:
            trial.report(val_loss, epochs_phase1 + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    logging.info("Tanítás kész! Legjobb modell: snake_best_full.pth")
    return best_val_loss
    

@torch.no_grad()
def validate(model, val_loader, crit_bin, crit_sp):
    model.eval()
    total_loss = 0.0
    bin_acc = sp_acc = 0.0
    for imgs, ven_lbl, sp_lbl in val_loader:
        # Move batch to GPU
        imgs = imgs.to(model.device, non_blocking=True)
        ven_lbl = ven_lbl.to(model.device, non_blocking=True)
        sp_lbl = sp_lbl.to(model.device, non_blocking=True)

        with torch.autocast(device_type=model.device.type):
            bin_logit, sp_logit = model(imgs)
            loss_bin = crit_bin(bin_logit.squeeze(), ven_lbl)
            loss_sp = crit_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp
        # Cast logits to FP32 for accurate metrics
        bin_logit = bin_logit.float()
        sp_logit = sp_logit.float()
        
        total_loss += loss.item()

        bin_acc += binary_accuracy(bin_logit, ven_lbl)
        sp_acc += species_accuracy(sp_logit, sp_lbl)

    return (total_loss / len(val_loader),
            bin_acc / len(val_loader),
            sp_acc / len(val_loader))