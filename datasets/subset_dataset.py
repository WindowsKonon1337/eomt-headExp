# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Wrapper to create train/val splits from a full dataset."""

from typing import Optional, List
import torch


class SubsetDataset(torch.utils.data.Dataset):
    """Wrapper that creates a subset of a dataset using specific indices."""
    
    def __init__(self, dataset: torch.utils.data.Dataset, indices: List[int], transforms=None):
        """
        Args:
            dataset: The full dataset to subset
            indices: List of indices to include in this subset
            transforms: Optional transforms to apply (overrides dataset's transforms)
        """
        self.dataset = dataset
        self.indices = indices
        self.original_transforms = dataset.transforms if hasattr(dataset, 'transforms') else None
        # Store the override transforms separately
        self.override_transforms = transforms
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        # Map the local index to the actual dataset index
        actual_idx = self.indices[idx]
        
        # Temporarily override dataset's transforms if needed
        if self.override_transforms is not None:
            original_ds_transforms = self.dataset.transforms
            self.dataset.transforms = self.override_transforms
            item = self.dataset[actual_idx]
            self.dataset.transforms = original_ds_transforms
        else:
            item = self.dataset[actual_idx]
        
        return item
    
    def close(self):
        """Close the underlying dataset if it has a close method."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
    
    def __del__(self):
        self.close()
