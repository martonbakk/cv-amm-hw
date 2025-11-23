DATA_SPLIT = 0.2
CLASS_NUM = 296
BATCH_SIZE = 48        # állítsd a GPU-d memóriájához (RTX 4090 → 64-96 is jó)
NUM_WORKERS = 2

# Fázisok learning rate-jei (ez a legjobb gyakorlattal egyezik)
LR_HEADS    = 1e-3
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05

EPOCHS_PHASE1 = 0   # csak a két fej tanul (backbone fagyasztva)
EPOCHS_PHASE2 = 10   # teljes modell finetuning

# Teljes augmentált mintaszám a training készletben
NUM_AUGMENTATIONS = 50000

# Optuna tuning idejére csökkentett epoch számok
OPTUNA_PHASE1_EPOCHS = 5
OPTUNA_PHASE2_EPOCHS = 10

IMAGE_ROOT     = "data/train_images_small"      # <-- IDE ÍRD A VALÓDI ÚTVONALAT
META_CSV       = "data/train_images_metadata2.csv"
LABEL_INFO_CSV = "data/venomous_status_metadata.csv"

