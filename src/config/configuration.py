DATA_SPLIT = 0.2
CLASS_NUM = 296
BATCH_SIZE = 32        # állítsd a GPU-d memóriájához (RTX 4090 → 64-96 is jó)
NUM_WORKERS = 8

# Fázisok learning rate-jei (ez a legjobb gyakorlattal egyezik)
LR_HEADS    = 1e-3
LR_BACKBONE = 3e-5
WEIGHT_DECAY = 0.05

EPOCHS_PHASE1 = 15   # csak a két fej tanul (backbone fagyasztva)
EPOCHS_PHASE2 = 30   # teljes modell finetuning

IMAGE_ROOT     = "data/train_images_small"      # <-- IDE ÍRD A VALÓDI ÚTVONALAT
META_CSV       = "data/train_images_metadata2.csv"
LABEL_INFO_CSV = "data/venomous_status_metadata.csv"