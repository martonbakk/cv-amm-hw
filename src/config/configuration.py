DATA_SPLIT = 0.2
CLASS_NUM = 296
BATCH_SIZE_1 = 40  # Adjust to your GPU memory (RTX 4090 → 64-96 is fine)
BATCH_SIZE_2 = 10  # Adjust to your GPU memory (RTX 4090 → 64-96 is fine)
NUM_WORKERS = 6
INPUT_SIZE = 672

# Learning rates for different phases (follows best practices)
LR_HEADS_BIN = 1e-3
LR_HEADS_MULTI = 1e-3
LR_BACKBONE = 1.35e-5
WEIGHT_DECAY = 0.00047
LABEL_SMOOTHING = 0.0009

EPOCHS_PHASE1 = 5  # Only the two heads learn (backbone frozen)
EPOCHS_PHASE2 = 30  # Full model fine-tuning

# Total number of augmented samples in the training set
NUM_AUGMENTATIONS = 200000

# Reduced epoch counts for Optuna tuning
OPTUNA_PHASE1_EPOCHS = 2
OPTUNA_PHASE2_EPOCHS = 10
OPTUNA_SUBSAMPLE_SIZE_TRAIN = 10000  # Sample size during Optuna tuning (for speed)
OPTUNA_SUBSAMPLE_SIZE_VAL = 2000  # Sample size during Optuna tuning (for speed)

IMAGE_ROOT = "data/train_images_large"  # <-- SET YOUR ACTUAL PATH HERE
META_CSV = "data/train_images_metadata.csv"
LABEL_INFO_CSV = "data/venomous_status_metadata.csv"

MODEL_NAME = "convnextv2_atto.fcmae"  # <-- BACKBONE
