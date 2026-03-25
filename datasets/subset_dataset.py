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
        self.transforms = transforms if transforms is not None else dataset.transforms
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int):
        # Map the local index to the actual dataset index
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx]
    
    def close(self):
        """Close the underlying dataset if it has a close method."""
        if hasattr(self.dataset, 'close'):
            self.dataset.close()
    
    def __del__(self):
        self.close()
