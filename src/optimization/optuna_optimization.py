import os
import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from src.data_loader.data_loader import DataLoader
from src.model.model import TwoHeadConvNeXtV2
from src.model.utils import train_model
import logging
from src.config.configuration import CLASS_NUM, EPOCHS_PHASE1, EPOCHS_PHASE2, NUM_AUGMENTATIONS
from src.data_loader.augmentation import Augmentor
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_optuna_tuning(
    data_loader: DataLoader,
    n_trials: int = 50,
    output_dir: str = ".",
    study_name: str = "snake_optimization",
    seed: int = 42,
    full_train_epochs: tuple = (EPOCHS_PHASE1, EPOCHS_PHASE2)
):
    """
    Runs full hyperparameter optimization pipeline with Optuna
    
    Args:
        data_loader: Initialized DataLoader instance
        n_trials: Number of optimization trials to run
        output_dir: Directory to save results and models
        study_name: Name for the Optuna study
        seed: Random seed for reproducibility
        full_train_epochs: Tuple of (phase1_epochs, phase2_epochs) for final training
        num_augmentations_final: Number of augmentations for final training
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "optuna.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting Optuna optimization with {n_trials} trials")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Study name: {study_name}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=2
        ),
        storage=f"sqlite:///{os.path.join(output_dir, 'optuna.db')}",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, data_loader),
        n_trials=n_trials,
        gc_after_trial=True
    )
    
    # Save study results
    study.trials_dataframe().to_csv(os.path.join(output_dir, "trials.csv"))
    
    # Print and save best results
    best_trial = study.best_trial
    logging.info("Best trial:")
    logging.info(f"  Value: {best_trial.value}")
    logging.info("  Params:")
    for key, value in best_trial.params.items():
        logging.info(f"    {key}: {value}")
    
    # Save best parameters
    params_path = os.path.join(output_dir, "best_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(best_trial.params, f)
    logging.info(f"Best parameters saved to {params_path}")
    
    # Train final model with best parameters
    logging.info("\nTraining final model with best parameters...")
    final_model = TwoHeadConvNeXtV2(
        num_multi_classes=CLASS_NUM,
        dropout=best_trial.params["dropout"]
    ).to(DEVICE)
    
    final_augmentor = Augmentor(
        num_augmentations=NUM_AUGMENTATIONS,
        center_n_transforms=best_trial.params["center_n_transforms"],
        center_magnitude=best_trial.params["center_magnitude"]
    )
    
    # Run full training with best parameters
    train_model(
        data_loader=data_loader,
        model=final_model,
        augmentor=final_augmentor,
        lr_heads=best_trial.params["lr_heads"],
        lr_backbone=best_trial.params["lr_backbone"],
        weight_decay=best_trial.params["weight_decay"],
        epochs_phase1=full_train_epochs[0],
        epochs_phase2=full_train_epochs[1],
        save_models=True
    )
    
    # Save final model with descriptive name
    final_model_path = os.path.join(output_dir, f"snake_final_{study_name}.pth")
    torch.save(final_model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    
    return best_trial, final_model_path

# Modified objective function to accept data_loader
def objective(trial: optuna.Trial, data_loader: DataLoader):
    # Sample hyperparameters
    lr_heads = trial.suggest_float("lr_heads", 1e-5, 1e-2, log=True)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    center_n_transforms = trial.suggest_int("center_n_transforms", 1, 5)
    center_magnitude = trial.suggest_int("center_magnitude", 5, 20)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Create model
    model = TwoHeadConvNeXtV2(
        num_multi_classes=CLASS_NUM,
        dropout=dropout
    ).to(DEVICE)

    # Create augmentor with reduced samples for tuning
    augmentor = Augmentor(
        num_augmentations=NUM_AUGMENTATIONS,
        center_n_transforms=center_n_transforms,
        center_magnitude=center_magnitude
    )

    # Run training with reduced epochs
    return train_model(
        data_loader=data_loader,
        model=model,
        augmentor=augmentor,
        lr_heads=lr_heads,
        lr_backbone=lr_backbone,
        weight_decay=weight_decay,
        epochs_phase1=3,   # Reduced for tuning
        epochs_phase2=6,   # Reduced for tuning
        save_models=False,
        trial=trial
    )