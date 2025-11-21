import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image
import os
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import DATA_SPLIT


@dataclass
class Sample:
    original_class: str
    original_venomous: bool
    predicted_class: Optional[int]
    predicted_venomous: Optional[bool]
    image: np.ndarray
    info: Dict


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
        self.data_set: List[Sample] = self.__load_data()

    def __load_data(self) -> List[Sample]:
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
            meta_data_df.iterrows(), total=len(meta_data_df), desc="Loading images"
        ):
            img_path = os.path.join(self.__image_data_set_path, row["image_path"])
            if not os.path.exists(img_path):
                logging.error(f"Image path {img_path} does not exist. Skipping...")
                continue

            original_class: str = row["class_id"]
            original_venomous: bool = bool(row["MIVS"])
            info = row.to_dict().copy()
            img = Image.open(img_path).convert("RGB")

            img = np.array(img)

            samples.append(
                Sample(
                    original_class=original_class,
                    original_venomous=original_venomous,
                    predicted_class=None,
                    predicted_venomous=None,
                    image=img,
                    info=info,
                )
            )
        return samples
