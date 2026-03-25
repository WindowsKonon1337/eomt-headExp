# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
from torch.utils.data import DataLoader
import json
import numpy as np
import zipfile

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset
from datasets.subset_dataset import SubsetDataset

# Strawberry dataset classes
CLASS_MAPPING = {
    0: 0,  # strawberry_ripe
    1: 1,  # strawberry_unripe
    2: 2,  # strawberry_half_ripe
    3: 3,  # peduncle
}


class StrawberryDataset(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 4,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
        train_split: float = 0.8,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.train_split = train_split

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        """
        Parse target masks for strawberry dataset.
        
        Args:
            target: RGB mask image where each pixel color represents an instance
            labels_by_id: Mapping from instance ID to class ID
            is_crowd_by_id: Mapping from instance ID to is_crowd flag
            
        Returns:
            Tuple of (masks, labels, is_crowd)
        """
        # Convert RGB to instance ID: R + G*256 + B*256^2
        if target.ndim == 3:
            target_id = target[0, :, :].long() + target[1, :, :].long() * 256 + target[2, :, :].long() * 256**2
        else:
            target_id = target.long()

        masks, labels, is_crowd = [], [], []

        for instance_id in target_id.unique():
            instance_id_val = instance_id.item()
            
            # Skip background (0)
            if instance_id_val == 0:
                continue
            
            if instance_id_val not in labels_by_id:
                continue

            cls_id = labels_by_id[instance_id_val]
            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target_id == instance_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id.get(instance_id_val, False))

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Load annotations to create labels_by_id mapping
        path_obj = Path(self.path)
        
        # Handle ZIP file case
        if path_obj.suffix.lower() == '.zip':
            if not path_obj.exists():
                raise FileNotFoundError(f"ZIP file not found at {path_obj}")
            
            # Read annotations from ZIP file
            try:
                with zipfile.ZipFile(path_obj, 'r') as zip_ref:
                    # Try to find annotations.json - it might be in a subfolder
                    all_files = zip_ref.namelist()
                    ann_candidates = [f for f in all_files if f.endswith('annotations.json')]
                    
                    if not ann_candidates:
                        raise FileNotFoundError(f"No annotations.json found in ZIP: {path_obj}")
                    
                    # Use the first one found (prefer root-level if it exists)
                    ann_path = ann_candidates[0]
                    
                    with zip_ref.open(ann_path, 'r') as f:
                        coco_data = json.load(f)
            except KeyError as e:
                raise FileNotFoundError(f"Annotations file not found inside ZIP: {path_obj}") from e
        else:
            # Handle directory case
            annotations_path = path_obj / "annotations.json"
            
            if not annotations_path.exists():
                raise FileNotFoundError(f"Annotations file not found at {annotations_path}")
            
            with open(annotations_path, 'r') as f:
                coco_data = json.load(f)
        
        # Build mappings
        labels_by_id = {}
        is_crowd_by_id = {}
        
        for ann in coco_data.get('annotations', []):
            instance_id = ann.get('instance_id', ann.get('id'))
            category_id = ann.get('category_id', 0)
            labels_by_id[instance_id] = category_id
            is_crowd_by_id[instance_id] = ann.get('iscrowd', False)
        
        # Get total number of images for split
        total_images = len(coco_data.get('images', []))
        split_idx = int(total_images * self.train_split)
        
        dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
            "img_folder_path_in_zip": Path("./images"),
            "target_folder_path_in_zip": Path("./masks"),
            "annotations_json_path_in_zip": Path("./annotations.json"),
        }
        
        # Check if path is a ZIP file
        zip_path = Path(self.path)
        if zip_path.suffix.lower() == '.zip':
            # Path is a ZIP file - we need to handle the internal structure
            # If ZIP contains a base folder, we need to add it to the paths
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                # Check if there's a common base folder
                first_level_dirs = set()
                for f in all_files:
                    parts = f.split('/')
                    if len(parts) > 1 and parts[0] and not parts[0].endswith('.json'):
                        first_level_dirs.add(parts[0])
                
                # If there's a single base folder containing everything, use it
                if len(first_level_dirs) == 1:
                    base_folder = list(first_level_dirs)[0]
                    dataset_kwargs["img_folder_path_in_zip"] = Path(base_folder) / "images"
                    dataset_kwargs["target_folder_path_in_zip"] = Path(base_folder) / "masks"
                    dataset_kwargs["annotations_json_path_in_zip"] = Path(base_folder) / "annotations.json"
            
            # Convert paths to use forward slashes for ZIP files
            for key in ["img_folder_path_in_zip", "target_folder_path_in_zip", "annotations_json_path_in_zip"]:
                dataset_kwargs[key] = Path(dataset_kwargs[key].as_posix())
        
        # Create the full dataset first with NO transforms (SubsetDataset will apply appropriate transforms)
        full_dataset = Dataset(
            zip_path=zip_path,
            transforms=None,  # No transforms on full dataset - SubsetDataset will apply them
            **dataset_kwargs,
        )
        
        # Calculate split indices
        total_images = len(full_dataset)
        split_idx = int(total_images * self.train_split)
        
        train_indices = list(range(0, split_idx))
        val_indices = list(range(split_idx, total_images))
        
        # Create train and val subsets with appropriate transforms
        self.train_dataset = SubsetDataset(
            full_dataset,
            train_indices,
            transforms=self.transforms,
        )
        
        # Val dataset uses no augmentation (only resizing)
        val_transforms = Transforms(
            img_size=self.transforms.img_size,
            color_jitter_enabled=False,
            scale_range=(1.0, 1.0),  # No scaling
        )
        
        self.val_dataset = SubsetDataset(
            full_dataset,
            val_indices,
            transforms=val_transforms,
        )
        
        print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val (split={self.train_split})")

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
