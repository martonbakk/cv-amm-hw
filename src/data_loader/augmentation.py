import logging
from typing import Optional, List
from src.data_loader.data_loader import Sample
import os
from dataclasses import dataclass
import numpy as np
import sys
import torch
from PIL import Image
import random
from dataclasses import replace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config.configuration import DATA_SPLIT

# Check for torchvision availability with fallback
TORCHVISION_AVAILABLE = False
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    logging.warning(
        "torchvision not installed. RandAugment will be unavailable. "
        "Install with: pip install torchvision"
    )

class SnakeAugmentor:
    """
    Static class for snake image augmentation using RandAugment.
    Designed for hyperparameter tuning with Optuna.
    """
    
    @staticmethod
    def randaugment(
        image: np.ndarray, 
        n_transforms: int = 2, 
        magnitude: int = 9,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply RandAugment transformations to a snake image.
        
        Args:
            image: Input image as numpy array (HWC format, uint8, [0-255])
            n_transforms: Number of sequential transformations to apply (Optuna parameter)
            magnitude: Intensity of transformations (0-30 scale) (Optuna parameter)
            seed: Random seed for reproducibility
        
        Returns:
            Augmented image as numpy array (same format as input)
        
        Raises:
            RuntimeError: If torchvision is not installed
            ValueError: If input parameters are invalid
        """
        # Dependency check
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError(
                "torchvision is required for RandAugment. Install with: pip install torchvision"
            )
        
        # Parameter validation
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError("Input image must be a uint8 numpy array")
        
        if n_transforms < 0:
            raise ValueError("n_transforms must be non-negative")
        
        if not (0 <= magnitude <= 30):
            raise ValueError("magnitude must be between 0 and 30 (inclusive)")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)  # Requires torch import if seed is used
        
        try:
            # Convert numpy array (HWC) to PIL Image
            pil_image = Image.fromarray(image)
            
            # Create RandAugment transform
            augmenter = transforms.RandAugment(
                num_ops=n_transforms,
                magnitude=magnitude,
                num_magnitude_bins=31  # Standard 0-30 magnitude scale
            )
            
            # Apply augmentation
            augmented_pil = augmenter(pil_image)
            
            # Convert back to numpy array
            return np.array(augmented_pil)
        
        except Exception as e:
            logging.error(f"Augmentation failed: {str(e)}")
            raise RuntimeError(f"Image augmentation error: {str(e)}") from e
    
    @staticmethod
    def get_optuna_search_space(trial) -> dict:
        """
        Define Optuna search space for RandAugment hyperparameters.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary with sampled hyperparameters
        """
        return {
            "n_transforms": trial.suggest_int("n_transforms", 1, 4),
            "magnitude": trial.suggest_int("magnitude", 5, 20)
        }

class Augmentor:
    """
    Class for augmenting a dataset of snake images using RandAugment.
    """
    def __init__(
        self, num_augmentations: int, center_n_transforms: int, center_magnitude: int, seed: Optional[int] = None
    ) -> None:
        self.num_augmentations = num_augmentations
        self.center_n_transforms = center_n_transforms
        self.center_magnitude = center_magnitude
        self.seed = seed
    def augment_dataset(
        self,
        train_data: List[Sample]
    ) -> List[Sample]:
        """
        Augments a dataset by generating new samples using RandAugment with parameters
        sampled around specified center values. Returns combined original and augmented
        samples in random order.
        
        Args:
            train_data: Original list of Sample objects
            num_augmentations: Number of new augmented samples to generate
            center_n_transforms: Center value for RandAugment n_transforms parameter
            center_magnitude: Center value for RandAugment magnitude parameter
            seed: Random seed for reproducibility
            
        Returns:
            Combined and shuffled list of original and augmented Sample objects
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        if self.num_augmentations <= 0:
            logging.info("No augmentations requested. Returning shuffled original data.")
            combined = train_data.copy()
            random.shuffle(combined)
            return combined
        
        if not train_data:
            logging.warning("Empty training data provided. Returning empty list.")
            return []
        
        augmented_samples = []
        for _ in range(self.num_augmentations):
            # Randomly select a sample to augment
            original_sample = random.choice(train_data)
            
            # Sample augmentation parameters around center values
            rng = np.random.default_rng()
            n_t = max(0, min(round(self.center_n_transforms + rng.normal(scale=self.center_n_transforms*0.5)), 10))
            mag = max(0, min(round(self.center_magnitude + rng.normal(scale=self.center_magnitude*0.5)), 30))
            
            # Apply augmentation with error fallback
            try:
                augmented_image = SnakeAugmentor.randaugment(
                    image=original_sample.image.copy(),
                    n_transforms=int(n_t),
                    magnitude=int(mag),
                    seed=self.seed
                )
            except Exception as e:
                logging.warning(f"Augmentation failed: {str(e)}. Using original image.")
                augmented_image = original_sample.image.copy()
            
            # Create augmented sample with metadata
            aug_info = {
                **original_sample.info,
                'augmented': True,
                'aug_params': {
                    'n_transforms': n_t,
                    'magnitude': mag
                }
            }
            
            augmented_samples.append(
                replace(
                    original_sample,
                    image=augmented_image,
                    predicted_class=None,
                    predicted_venomous=None,
                    info=aug_info
                )
            )
        
        # Combine and shuffle
        combined = train_data + augmented_samples
        random.shuffle(combined)
        logging.info(f"Augmentation complete: {len(train_data)} original + "
                    f"{len(augmented_samples)} augmented = {len(combined)} total samples")
        return combined    
