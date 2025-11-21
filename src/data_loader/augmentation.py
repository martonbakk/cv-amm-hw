import logging
from typing import Optional
import os
from dataclasses import dataclass
import numpy as np
import sys
import torch
from PIL import Image

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
            "magnitude": trial.suggest_int("magnitude", 5, 15)
        }