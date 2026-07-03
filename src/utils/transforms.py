import torch
import torch.nn as nn
from torchvision import transforms
import logging

log = logging.getLogger(__name__)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Expects tensor in [0, 1] range preferably
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transforms(config, split='train'):
    """
    Returns the appropriate transform pipeline based on stage and config.
    stage: 'train' or 'val'/'test'
    """
    img_size = getattr(config, 'img_size', 224)
    aug_mode = getattr(config, 'aug_mode', 'soft') # 'soft', 'hard

    # Validation / Test (Deterministic)
    if split != 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            # Note: No ToTensor() needed because _load_mri returns a Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Training Augmentations
    if aug_mode == 'hard':
        # log it
        log.info("Using hard augmentation mode for training transforms.")
        
        return transforms.Compose([
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), antialias=True),
            transforms.RandomRotation(degrees=20),        
            transforms.ColorJitter(brightness=0.4, contrast=0.4),          
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2), 
            transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.3),            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    else: # Default 'soft'
        log.info("Using soft augmentation mode for training transforms.")
        return transforms.Compose([
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0), antialias=True),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])