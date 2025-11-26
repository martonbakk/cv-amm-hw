from typing import List, Optional
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
    LR_HEADS_BIN,
    LR_HEADS_MULTI,
    BATCH_SIZE_PHASE_1,
    BATCH_SIZE_PHASE_2,
    OPTUNA_SUBSAMPLE_SIZE_TRAIN,
    OPTUNA_SUBSAMPLE_SIZE_VAL,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
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
            if sample.info.get("augmented", False):
                # Get source image path directly from metadata
                source_img_path = sample.info["aug_params"]["source_image_path"]
                full_path = os.path.join(self.image_root, source_img_path)

                # Load source image ON DEMAND
                img = Image.open(full_path).convert("RGB")

                # Apply augmentation
                try:
                    augmented_img = SnakeAugmentor.randaugment(
                        image=img,
                        n_transforms=sample.info.get("aug_params", {}).get(
                            "n_transforms", 2
                        ),
                        magnitude=sample.info.get("aug_params", {}).get(
                            "magnitude", 10
                        ),
                    )
                except Exception as e:
                    logging.info(f"Augmentation failed: {str(e)}. Using original.")
                    logging.info("Augmentation failed, using original image.")
                finally:
                    img = basic_transform(augmented_img)
            else:
                # Regular sample - load normally
                img_path = sample.info.get("image_path", None)

                if img_path is None:
                    raise ValueError("Image_path not found in Sample.info!")
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
        persistent_workers=True,
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
    lr_heads_bin: float = LR_HEADS_BIN,
    lr_heads_multi: float = LR_HEADS_MULTI,
    lr_backbone: float = LR_BACKBONE,
    weight_decay: float = WEIGHT_DECAY,
    label_smoothing: float = LABEL_SMOOTHING,
    epochs_phase1: int = EPOCHS_PHASE1,
    epochs_phase2: int = EPOCHS_PHASE2,
    save_models: bool = True,
    trial: Optional[Trial] = None,
):

    train_samples: ListDataset = data_loader.get_training_set()
    val_samples: ListDataset = data_loader.get_validation_set()

    if augmentor is not None:
        augmented_samples = create_augmented_samples(train_samples, augmentor)
        combined_samples = list(train_samples) + augmented_samples
        random.shuffle(combined_samples)
        train_samples = ListDataset(combined_samples)
        logging.info(f"Created {len(augmented_samples)} virtual augmented samples")

    # Subsample if optuna is running the function
    if trial is not None:
        train_samples = ListDataset(train_samples[:OPTUNA_SUBSAMPLE_SIZE_TRAIN])
        val_samples = ListDataset(val_samples[:OPTUNA_SUBSAMPLE_SIZE_VAL])
        logging.info("Optuna tuning: Using reduced dataset for faster training")

    train_loader: TorchDataLoader = create_torch_loader(
        train_samples, BATCH_SIZE_PHASE_1, shuffle=True
    )
    val_loader: TorchDataLoader = create_torch_loader(
        val_samples, BATCH_SIZE_PHASE_1, shuffle=False
    )

    logging.info("PHASE 1: Training only the heads (backbone frozen)")
    model.freeze_backbone()
    optimizer_bin = optim.AdamW(
        model.binary_head.parameters(), lr=lr_heads_bin, weight_decay=weight_decay
    )
    optimizer_sp = optim.AdamW(
        model.species_head.parameters(), lr=lr_heads_multi, weight_decay=weight_decay
    )
    criterion_bin = nn.BCEWithLogitsLoss()  # binary loss
    criterion_sp = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )  # multi-class loss
    scheduler_bin = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_bin, patience=3, factor=0.5
    )  # Must come AFTER the optimizer
    scheduler_sp = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sp, patience=3, factor=0.5
    )  # Must come AFTER the optimizer
    scaler = torch.GradScaler()  # Define gradient scaler for mixed precision
    run_training_phase_heads(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_bin=optimizer_bin,
        optimizer_sp=optimizer_sp,
        scheduler_bin=scheduler_bin,
        scheduler_sp=scheduler_sp,
        criterion_bin=criterion_bin,
        criterion_sp=criterion_sp,
        scaler=scaler,
        epochs=epochs_phase1,
        save_models=save_models,
        trial=trial,
        start_epoch_offset=0,
    )

    train_loader: TorchDataLoader = create_torch_loader(
        train_samples, BATCH_SIZE_PHASE_2, shuffle=True
    )
    val_loader: TorchDataLoader = create_torch_loader(
        val_samples, BATCH_SIZE_PHASE_2, shuffle=False
    )

    logging.info("=== PHASE 2: Full model fine-tuning ===")
    model.unfreeze_backbone()
    best_val_loss = float("inf")
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.binary_head.parameters(), "lr": lr_heads_bin},
            {"params": model.species_head.parameters(), "lr": lr_heads_multi},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_phase2
    )  # Must come AFTER the optimizer
    best_val_loss = run_training_phase_full(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion_bin=criterion_bin,
        criterion_sp=criterion_sp,
        scaler=scaler,
        epochs=epochs_phase2,
        save_models=save_models,
        trial=trial,
        start_epoch_offset=epochs_phase1
    )
    return best_val_loss


def run_training_phase_heads(
    model: TwoHeadConvNeXtV2,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    optimizer_bin: optim.AdamW,
    optimizer_sp: optim.AdamW,
    scheduler_bin: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_sp: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion_bin: nn.BCEWithLogitsLoss,
    criterion_sp: nn.CrossEntropyLoss,
    scaler: torch.GradScaler,
    epochs: int,
    save_models: bool = False,
    trial: Optional[optuna.trial.Trial] = None,
    start_epoch_offset: int = 0,
):
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase1 Epoch {epoch}"):
            imgs = imgs.to(model.device, non_blocking=True)
            ven_lbl = ven_lbl.to(model.device, non_blocking=True)
            sp_lbl = sp_lbl.to(model.device, non_blocking=True)

            optimizer_bin.zero_grad()
            optimizer_sp.zero_grad()
            with torch.autocast(device_type=model.device.type):
                bin_logit, sp_logit = model(imgs)
                loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)

            # Backward + optimizer step separately
            scaler.scale(loss_bin).backward(retain_graph=True)
            scaler.step(optimizer_bin)
            scaler.update()

            scaler.scale(loss_sp).backward()
            scaler.step(optimizer_sp)
            scaler.update()

            epoch_loss += loss_bin.item() + loss_sp.item()

        # Validation
        val_loss, val_bin_acc, val_sp_acc = validate(
            model, val_loader, criterion_bin, criterion_sp
        )
        scheduler_bin.step(val_loss)
        scheduler_sp.step(val_loss)

        logging.info(
            f"Phase1 | Epoch {epoch} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
            f"ValLoss {val_loss:.4f} | BinAcc {val_bin_acc:.2f}% | MultiAcc {val_sp_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_models:
                torch.save(model.state_dict(), "phase1_best.pth")
                logging.info("New best model saved!")

        if trial:
            trial.report(val_loss, start_epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def run_training_phase_full(
    model: TwoHeadConvNeXtV2,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    optimizer: optim.AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    criterion_bin: nn.BCEWithLogitsLoss,
    criterion_sp: nn.CrossEntropyLoss,
    scaler: torch.GradScaler,
    epochs: int,
    save_models: bool = False,
    trial: Optional[optuna.trial.Trial] = None,
    start_epoch_offset: int = 0,
):
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, ven_lbl, sp_lbl in tqdm(train_loader, desc=f"Phase2 Epoch {epoch}"):
            imgs = imgs.to(model.device, non_blocking=True)
            ven_lbl = ven_lbl.to(model.device, non_blocking=True)
            sp_lbl = sp_lbl.to(model.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=model.device.type):
                bin_logit, sp_logit = model(imgs)
                loss_bin = criterion_bin(bin_logit.squeeze(), ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)
                loss = loss_bin + loss_sp

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        val_loss, val_bin_acc, val_sp_acc = validate(
            model, val_loader, criterion_bin, criterion_sp
        )
        scheduler.step()

        logging.info(
            f"Phase2 | Epoch {epoch} | TrainLoss {epoch_loss/len(train_loader):.4f} | "
            f"ValLoss {val_loss:.4f} | BinAcc {val_bin_acc:.2f}% | MultiAcc {val_sp_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_models:
                torch.save(model.state_dict(), "phase2_best.pth")
                logging.info("New best model saved!")

        if trial:
            trial.report(val_loss, start_epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def create_augmented_samples(
    train_samples: ListDataset, augmentor: Augmentor
) -> List[Sample]:
    """
    Generate augmented samples based on the provided augmentor.

    Args:
        train_samples (ListDataset): The original training dataset.
        augmentor (Augmentor): Augmentation configuration.

    Returns:
        List[Sample]: List of augmented Sample instances.
    """
    augmented_samples = []

    for _ in range(augmentor.num_augmentations):
        # Randomly select a source sample
        source_sample = random.choice(train_samples)

        # Sample augmentation parameters
        rng = np.random.default_rng()
        n_t = max(
            0,
            min(
                round(
                    augmentor.center_n_transforms
                    + rng.normal(scale=augmentor.center_n_transforms * 0.5)
                ),
                10,
            ),
        )
        mag = max(
            0,
            min(
                round(
                    augmentor.center_magnitude
                    + rng.normal(scale=augmentor.center_magnitude * 0.5)
                ),
                30,
            ),
        )

        # Create augmented sample metadata
        aug_info = {
            **source_sample.info,
            "augmented": True,
            "aug_params": {
                "n_transforms": n_t,
                "magnitude": mag,
                "source_image_path": source_sample.info["image_path"],
            },
        }

        augmented_samples.append(
            Sample(
                original_class=source_sample.original_class,
                original_venomous=source_sample.original_venomous,
                predicted_class=None,
                predicted_venomous=None,
                info=aug_info,
            )
        )

    return augmented_samples


@torch.no_grad()
def validate(model, val_loader, crit_bin, crit_sp):
    model.eval()
    total_loss = 0.0
    bin_acc = sp_acc = 0.0

    for imgs, ven_lbl, sp_lbl in val_loader:
        imgs = imgs.to(model.device, non_blocking=True)
        ven_lbl = ven_lbl.to(model.device, non_blocking=True)
        sp_lbl = sp_lbl.to(model.device, non_blocking=True)

        with torch.autocast(
            device_type=model.device.type
        ):  # Cast logits to FP32 for accurate metrics
            bin_logit, sp_logit = model(imgs)
            loss_bin = crit_bin(bin_logit.squeeze(), ven_lbl)
            loss_sp = crit_sp(sp_logit, sp_lbl)
            loss = loss_bin + loss_sp

        bin_logit = bin_logit.float()
        sp_logit = sp_logit.float()
        total_loss += loss.item()

        bin_acc += binary_accuracy(bin_logit, ven_lbl)
        sp_acc += species_accuracy(sp_logit, sp_lbl)

    return (
        total_loss / len(val_loader),
        bin_acc / len(val_loader),
        sp_acc / len(val_loader),
    )
