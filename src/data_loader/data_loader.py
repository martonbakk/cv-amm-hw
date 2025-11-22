import logging
import random
from typing import Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys, os
import torch as th
import logging
from torchvision import transforms
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import DATA_SPLIT


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Egyszerű átméretezés 224×224-re + ToTensor + normalizálás (train és val ugyanaz!)
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # ConvNeXtV2 224×224-et vár
    transforms.ToTensor(),          # [H, W, C] → [C, H, W] + 0-1 float
    normalize,                       # ImageNet statok
])


@dataclass
class Sample:
    original_class: str
    original_venomous: bool
    predicted_class: Optional[int]
    predicted_venomous: Optional[bool]
    image: Optional[th.Tensor]
    info: Dict

class ListDataset(th.utils.data.Dataset):
    """Small wrapper to present a plain list as a torch Dataset for type-checkers."""
    def __init__(self, items):
        # ensure we have a concrete list (None handled by caller)
        self.__items = list(items)

    def __len__(self):
        return len(self.__items)

    def __getitem__(self, idx):
        return self.__items[idx]
    
    @property
    def items(self):
        return self.__items

class DataLoader:
    def __init__(
        self, image_data_set_path: str, meta_data_path: str, label_info_path: str
    ) -> None:
        """Initializes the DataLoader with the given dataset path.
        Args:
            image_data_set_path (str): Path to the dataset directory.
            meta_data_path (str): csv file containing meta data information.
            label_meta_data_path (str): csv file containing the labels.
        """
        logging.info("Initializing DataLoader...")
        self.__image_data_set_path: str = image_data_set_path
        self.__meta_data_path: str = meta_data_path
        self.__label_info_path: str = label_info_path
        all_samples: ListDataset = self.__load_data()
        
        unique_old_ids = sorted(set(s.original_class for s in all_samples))
        old_to_new = {old: new for new, old in enumerate(unique_old_ids)}

        for sample in all_samples:
            sample.original_class = old_to_new[sample.original_class] # pyright: ignore[reportAttributeAccessIssue]

        random.Random(42).shuffle(all_samples.items)
        labels_for_stratify = [s.original_class for s in all_samples]
        train_samples, val_samples = train_test_split(
            all_samples, test_size=DATA_SPLIT, stratify=labels_for_stratify, random_state=42
        )
        logging.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

        self.__training_set: ListDataset = train_samples
        self.__validation_set: ListDataset = val_samples

    def __load_data(self) -> ListDataset:
        """Loads the dataset from the specified path.
        Returns:
            Tuple[List[Sample], List[Sample]]: A tuple containing the training and validation datasets.

        Raises:
            FileNotFoundError: If the dataset path does not exist.
        """
        logging.info("Checking paths...")
        if not os.path.exists(self.__meta_data_path):
            raise FileNotFoundError(
                f"Info Metadata path {self.__meta_data_path} does not exist."
            )

        if not os.path.exists(self.__label_info_path):
            raise FileNotFoundError(
                f"Label info path {self.__label_info_path} does not exist."
            )

        if not os.path.isdir(self.__image_data_set_path):
            raise FileNotFoundError(
                f"Image path {self.__image_data_set_path} does not exist."
            )

        logging.info(f"Loading metadata from {self.__meta_data_path}...")
        meta_data_df = pd.read_csv(self.__meta_data_path, index_col=0).reset_index(
            drop=True
        )
        logging.debug(f"Metadata columns: {meta_data_df.columns.tolist()}")

        logging.info(f"Loading label info from {self.__label_info_path}...")
        label_info_df = pd.read_csv(self.__label_info_path, index_col=0).reset_index(
            drop=True
        )
        logging.debug(f"Label info columns: {label_info_df.columns.tolist()}")
        meta_data_df = meta_data_df.merge(label_info_df, on="class_id")
        logging.debug(f"Merged Metadata columns: {meta_data_df.columns.tolist()}")

        samples: List[Sample] = []
        logging.info(f"Loading image data from {self.__image_data_set_path}...")

        for _, row in tqdm(
            meta_data_df.iterrows(), total=len(meta_data_df), desc="Loading metadata"
        ):
            img_path = os.path.join(self.__image_data_set_path, row["image_path"])
            if not os.path.exists(img_path):
                logging.error(f"Image path {img_path} does not exist. Skipping...")
                continue

            original_class: str = row["class_id"]
            original_venomous: bool = bool(row["MIVS"])
            info = row.to_dict().copy()

            samples.append(
                Sample(
                    original_class=original_class,
                    original_venomous=original_venomous,
                    predicted_class=None,
                    predicted_venomous=None,
                    image=None,
                    info=info,
                )
            )
        return ListDataset(samples)

    def get_training_set(self) -> Optional[ListDataset]:
        return self.__training_set

    def get_validation_set(self) -> Optional[ListDataset]:
        return self.__validation_set
    
    @staticmethod
    def load_image(image_path: str) -> th.Tensor:
        img = Image.open(image_path).convert("RGB")
        img = basic_transform(img)          # csak resize + ToTensor + normalize
        return img
