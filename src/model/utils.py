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
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import optuna
from optuna.trial import Trial
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import (
    IMAGE_ROOT,
    NUM_WORKERS,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_BACKBONE,
    LR_HEADS_BIN,
    LR_HEADS_MULTI,
    BATCH_SIZE_1,
    BATCH_SIZE_2,
    OPTUNA_SUBSAMPLE_SIZE_TRAIN,
    OPTUNA_SUBSAMPLE_SIZE_VAL,
    WEIGHT_DECAY,
    LABEL_SMOOTHING,
    CLASS_NUM,
)
from src.data_loader.data_loader import ListDataset, Sample, basic_transform
from src.data_loader.augmentation import SnakeAugmentor, Augmentor
from src.data_loader.data_loader import DataLoader
from src.model.model import TwoHeadConvNeXtV2


def get_class_weights(train_samples, num_classes=CLASS_NUM):
    labels = [s.original_class for s in train_samples]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float32)


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
        train_samples, BATCH_SIZE_1, shuffle=True
    )
    val_loader: TorchDataLoader = create_torch_loader(
        val_samples, BATCH_SIZE_1, shuffle=False
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
        optimizer_bin, patience=2, factor=0.5
    )  # Must come AFTER the optimizer
    scheduler_sp = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sp, patience=2, factor=0.5
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
        train_samples, BATCH_SIZE_2, shuffle=True
    )
    val_loader: TorchDataLoader = create_torch_loader(
        val_samples, BATCH_SIZE_2, shuffle=False
    )

    logging.info("PHASE 2: Full model fine-tuning")
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
        start_epoch_offset=0,
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
                loss_bin = criterion_bin(bin_logit, ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)
                loss = loss_bin + loss_sp

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer_bin)
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

        if save_models and val_loss < best_val_loss:
            best_val_loss = val_loss
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
                loss_bin = criterion_bin(bin_logit, ven_lbl)
                loss_sp = criterion_sp(sp_logit, sp_lbl)
                loss = loss_bin + loss_sp

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        if save_models and val_loss < best_val_loss:
            best_val_loss = val_loss
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
    rng = np.random.default_rng()

    # Get class weights and convert to sample probabilities
    class_weights = get_class_weights(train_samples, CLASS_NUM)
    sample_weights = np.array(
        [class_weights[sample.original_class].item() for sample in train_samples]
    )

    # Convert to sampling probabilities (higher weight = higher probability)
    sample_probs = sample_weights / sample_weights.sum()
    logging.info(
        f"Class-balanced augmentation: "
        f"Rarest class weight={class_weights.max().item():.1f}, "
        f"Most common weight={class_weights.min().item():.1f}"
    )

    for _ in range(augmentor.num_augmentations):
        # Randomly select a source sample with p based on class weights
        source_idx = np.random.choice(len(train_samples), p=sample_probs)
        source_sample = train_samples[source_idx]

        # Sample augmentation parameters
        n_t = max(
            0,
            min(
                round(
                    augmentor.center_n_transforms
                    + rng.normal(scale=augmentor.center_n_transforms * 0.2)
                ),
                10,
            ),
        )
        mag = max(
            0,
            min(
                round(
                    augmentor.center_magnitude
                    + rng.normal(scale=augmentor.center_magnitude * 0.2)
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
def validate(
    model: TwoHeadConvNeXtV2,
    val_loader: TorchDataLoader,
    crit_bin: torch.nn.Module,
    crit_sp: torch.nn.Module,
) -> Tuple[float, float, float]:
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
            loss_bin = crit_bin(bin_logit, ven_lbl)
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


@torch.no_grad()
def format_pred(
    model: TwoHeadConvNeXtV2, data_loader: TorchDataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    model.eval()

    all_venom_true = []
    all_venom_pred = []
    all_species_true = []
    all_species_pred = []

    pbar = tqdm(data_loader, desc="Running Inference")

    for imgs, ven_lbl, sp_lbl in pbar:
        imgs = imgs.to(model.device, non_blocking=True)

        with torch.autocast(device_type=model.device.type):
            bin_logits, sp_logits = model(imgs)

        preds_bin = torch.sigmoid(bin_logits).round().cpu().numpy()

        preds_species = torch.argmax(sp_logits, dim=1).cpu().numpy()

        all_venom_true.extend(ven_lbl.numpy())
        all_venom_pred.extend(preds_bin)
        all_species_true.extend(sp_lbl.numpy())
        all_species_pred.extend(preds_species)

    return (
        np.array(all_venom_true),
        np.array(all_venom_pred),
        np.array(all_species_true),
        np.array(all_species_pred),
    )


def compute_performance_metrics(
    venom_true: np.ndarray,
    venom_pred: np.ndarray,
    species_true: np.ndarray,
    species_pred: np.ndarray,
) -> Dict[str, float | int | np.ndarray]:
    # Macro Averaged F1 (Species)
    f1_macro = float(f1_score(species_true, species_pred, average="macro"))

    # Accuracy (Species)
    acc_species = float(accuracy_score(species_true, species_pred))

    # Binary Accuracy (Venomous accuracy)
    acc_binary = float(accuracy_score(venom_true, venom_pred))

    # Venomous Mistake Stats (Confusion Matrix)
    try:
        tn, fp, fn, tp = confusion_matrix(venom_true, venom_pred).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    except ValueError:
        # Fallback if batch contains only one class
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "acc_species": acc_species,
        "f1_macro": f1_macro,
        "acc_binary": acc_binary,
        "venom_true_pos": int(tp),
        "venom_true_neg": int(tn),
        "venom_false_pos": int(fp),
        "venom_false_neg": int(fn),
    }


def full_evaluation(
    model: TwoHeadConvNeXtV2, data_loader: DataLoader
) -> Dict[str, float | np.ndarray | int]:
    """
    Wrapper function to run inference and calculate metrics in one go.
    """
    val_loader = create_torch_loader(
        data_loader.get_validation_set(), BATCH_SIZE_2, shuffle=False
    )

    v_true, v_pred, s_true, s_pred = format_pred(model, val_loader)

    metrics = compute_performance_metrics(v_true, v_pred, s_true, s_pred)

    logging.info("EVALUATION RESULTS")
    logging.info(f"Species Macro F1: {metrics['f1_macro']:.4f}")
    logging.info(f"Species Accuracy: {metrics['acc_species']:.4f}")
    logging.info(f"Venom Binary Acc: {metrics['acc_binary']:.4f}")
    logging.info(f"Venom Mistakes (FN - Dangerous): {metrics['venom_false_neg']}")

    return metrics


def plot_snake_metrics(metrics):
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    metric_names = ["Species Accuracy", "Macro F1", "Venom Accuracy"]
    metric_values = [metrics["acc_species"], metrics["f1_macro"], metrics["acc_binary"]]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.8)

    ax1.set_ylim(0, 1.1)
    ax1.set_title("Modell Általános Teljesítménye", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Érték (0-1)", fontsize=12)

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    cm = np.array(
        [
            [metrics["venom_true_neg"], metrics["venom_false_pos"]],
            [metrics["venom_false_neg"], metrics["venom_true_pos"]],
        ]
    )

    group_names = [
        "TN\n(Helyes Ártalmatlan)",
        "FP\n(Téves Riasztás)",
        "FN\n(VESZÉLYES HIBA!)",
        "TP\n(Helyes Mérges)",
    ]

    group_counts = [f"{value:0.0f}" for value in cm.flatten()]

    row_sums = cm.sum(axis=1, keepdims=True)

    row_sums[row_sums == 0] = 1
    percentages = cm / row_sums
    group_percentages = [f"({value:.1%})" for value in percentages.flatten()]

    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax2,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
        square=True,
    )

    ax2.set_xlabel("Modell Tippje", fontsize=12)
    ax2.set_ylabel("Valóság", fontsize=12)
    ax2.set_xticklabels(["Ártalmatlan", "Mérges"])
    ax2.set_yticklabels(["Ártalmatlan", "Mérges"])
    ax2.set_title(
        "Mérgesség Detektálás (Konfúziós Mátrix)", fontsize=14, fontweight="bold"
    )

    rect = Rectangle((0, 1), 1, 1, fill=False, edgecolor="red", lw=4, clip_on=False)
    ax2.add_patch(rect)

    plt.tight_layout()
    plt.show()
