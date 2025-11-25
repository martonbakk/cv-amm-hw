from typing import Optional
import torch
from PIL import Image
import torchvision.transforms as transforms

class SnakeAugmentor:
    """
    Static class for snake image augmentation using RandAugment.
    Designed for hyperparameter tuning with Optuna.
    """
    
    @staticmethod
    def randaugment(
        image: Image.Image, 
        n_transforms: int = 2, 
        magnitude: int = 9,
        seed: Optional[int] = None
    ) -> Image.Image:
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
        
        if n_transforms < 0:
            raise ValueError("n_transforms must be non-negative")
        
        if not (0 <= magnitude <= 30):
            raise ValueError("magnitude must be between 0 and 30 (inclusive)")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)  # Requires torch import if seed is used
        
        # Create RandAugment transform
        augmenter = transforms.RandAugment(
            num_ops=n_transforms,
            magnitude=magnitude,
            num_magnitude_bins=31  # Standard 0-30 magnitude scale
        )
        
        # Apply augmentation
        augmented_pil = augmenter(image)
        return augmented_pil

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
