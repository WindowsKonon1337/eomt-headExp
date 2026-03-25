# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
# Adapted for Strawberry Dataset


from pathlib import Path
from typing import Union, Optional, Callable
from torch.utils.data import DataLoader
import json
import zipfile
import tempfile
import torch
from PIL import Image
from io import BytesIO
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

# Strawberry dataset class mapping (4 classes)
CLASS_MAPPING = {
    0: 0,  # strawberry_ripe
    1: 1,  # strawberry_unripe
    2: 2,  # strawberry_half_ripe
    3: 3,  # peduncle
}


class StrawberryDataset(Dataset):
    """
    Custom Dataset for strawberry data where masks are stored flat but images may be split.
    Matches images and masks by filename, ignoring subdirectories.
    """
    def __init__(
        self,
        zip_path,
        img_suffix,
        target_parser,
        check_empty_targets,
        transforms=None,
        only_annotations_json=False,
        target_suffix=None,
        stuff_classes=None,
        img_stem_suffix="",
        target_stem_suffix="",
        target_zip_path=None,
        target_zip_path_in_zip=None,
        target_instance_zip_path=None,
        img_folder_path_in_zip=Path("./"),
        target_folder_path_in_zip=Path("./"),
        target_instance_folder_path_in_zip=Path("./"),
        annotations_json_path_in_zip=None,
        flat_masks=False,
    ):
        from datasets.dataset import DirectoryZipAdapter
        
        if flat_masks:
            # Initialize parent class attributes manually
            self.zip_path = zip_path
            self.target_parser = target_parser
            self.transforms = transforms
            self.only_annotations_json = only_annotations_json
            self.stuff_classes = stuff_classes
            self.target_zip_path = target_zip_path
            self.target_zip_path_in_zip = target_zip_path_in_zip
            self.target_instance_zip_path = target_instance_zip_path
            self.target_folder_path_in_zip = target_folder_path_in_zip
            
            self.zip = None
            self.target_zip = None
            self.target_instance_zip = None
            
            # Load annotation data for labels_by_id
            self.labels_by_id = {}
            self.polygons_by_id = {}
            self.is_crowd_by_id = {}
            
            if annotations_json_path_in_zip is not None:
                anno_path = target_zip_path or zip_path
                path_obj = Path(anno_path)
                
                if path_obj.is_dir():
                    anno_zip = DirectoryZipAdapter(path_obj)
                else:
                    anno_zip = zipfile.ZipFile(anno_path)
                
                with anno_zip:
                    with anno_zip.open(annotations_json_path_in_zip.as_posix(), "r") as f:
                        annotation_data = json.load(f)
                
                # Build instance_id -> category_id mapping
                for ann in annotation_data.get('annotations', []):
                    instance_id = ann.get('instance_id', ann.get('id'))
                    self.labels_by_id[instance_id] = ann.get('category_id', 0)
                    self.is_crowd_by_id[instance_id] = ann.get('iscrowd', False)
            
            # Load image and mask filenames
            self.imgs = []
            self.targets = []
            self.targets_instance = []
            
            # Open ZIPs to list files
            if Path(zip_path).is_dir():
                img_zip = DirectoryZipAdapter(Path(zip_path))
            else:
                img_zip = zipfile.ZipFile(zip_path)
            
            try:
                img_files = [f.as_posix() if isinstance(f, Path) else f for f in img_zip.namelist()]
                
                # Get all PNG files from images folder
                image_filenames = {}  # maps stem to full path
                for img_file in img_files:
                    if 'images/' not in img_file:
                        continue
                    if not img_file.endswith('.png'):
                        continue
                    
                    # Extract filename (just the name, no path)
                    file_name = Path(img_file).name
                    stem = Path(file_name).stem
                    image_filenames[stem] = img_file
                
                # Get all mask files (should be flat)
                if Path(zip_path).is_dir():
                    target_zip = DirectoryZipAdapter(Path(zip_path))
                else:
                    target_zip = zipfile.ZipFile(zip_path)
                
                try:
                    target_files = target_zip.namelist()
                    mask_filenames = {}  # maps stem to full path
                    
                    for mask_file in target_files:
                        if 'masks/' not in mask_file:
                            continue
                        if not mask_file.endswith('.png'):
                            continue
                        
                        file_name = Path(mask_file).name
                        stem = Path(file_name).stem
                        # Handle case where mask might be in subdirectory
                        # But we prefer the flat version
                        if stem not in mask_filenames or '/' not in mask_file.replace('masks/', ''):
                            mask_filenames[stem] = mask_file
                    
                    # Match images to masks by stem
                    for stem, img_path in sorted(image_filenames.items()):
                        if stem in mask_filenames:
                            mask_path = mask_filenames[stem]
                            self.imgs.append(img_path)
                            self.targets.append(mask_path)
                            
                finally:
                    if hasattr(target_zip, 'close'):
                        target_zip.close()
                        
            finally:
                if hasattr(img_zip, 'close'):
                    img_zip.close()
        else:
            # Use parent class for non-flat case
            super().__init__(
                zip_path=zip_path,
                img_suffix=img_suffix,
                target_parser=target_parser,
                check_empty_targets=check_empty_targets,
                transforms=transforms,
                only_annotations_json=only_annotations_json,
                target_suffix=target_suffix,
                stuff_classes=stuff_classes,
                img_stem_suffix=img_stem_suffix,
                target_stem_suffix=target_stem_suffix,
                target_zip_path=target_zip_path,
                target_zip_path_in_zip=target_zip_path_in_zip,
                target_instance_zip_path=target_instance_zip_path,
                img_folder_path_in_zip=img_folder_path_in_zip,
                target_folder_path_in_zip=target_folder_path_in_zip,
                target_instance_folder_path_in_zip=target_instance_folder_path_in_zip,
                annotations_json_path_in_zip=annotations_json_path_in_zip,
            )





class StrawberryPanoptic(LightningDataModule):
    """
    Data module for Strawberry Dataset with instance and panoptic segmentation.
    Supports ZIP file storage of images and masks.
    """
    def __init__(
        self,
        path,
        stuff_classes: list[int] = None,
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
        self.stuff_classes = stuff_classes or []

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
                   Shape: (C, H, W) where C=3 (R, G, B channels)
            labels_by_id: Mapping from instance ID to class ID
            is_crowd_by_id: Mapping from instance ID to is_crowd flag
            
        Returns:
            Tuple of (masks, labels, is_crowd) lists
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
            
            # Skip instances not in annotations
            if instance_id_val not in labels_by_id:
                # For missing instances, try to infer class from surroundings or skip
                # For now, we'll assign them to the first available class if no label exists
                # This allows training even with incomplete annotations
                if not labels_by_id:
                    cls_id = 0  # Default to first class
                else:
                    continue  # Skip if we have annotations but this instance isn't in them

            else:
                cls_id = labels_by_id[instance_id_val]
            
            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target_id == instance_id)
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id.get(instance_id_val, False))

        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        """
        Setup the strawberry dataset for training/validation.
        
        Loads annotations and creates datasets for image masks stored in ZIP or directory format.
        Handles cases where masks are not split by train/val subdirectories.
        """
        # Load annotations to create labels_by_id mapping
        path_obj = Path(self.path)
        
        # Handle ZIP file case
        if path_obj.suffix.lower() == '.zip':
            if not path_obj.exists():
                raise FileNotFoundError(f"ZIP file not found at {path_obj}")
            
            # Read annotations from ZIP file
            annotations_relpath = 'annotations.json'
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
        
        # Build mappings from instance_id to class_id
        labels_by_id = {}
        is_crowd_by_id = {}
        
        for ann in coco_data.get('annotations', []):
            instance_id = ann.get('instance_id', ann.get('id'))
            category_id = ann.get('category_id', 0)
            labels_by_id[instance_id] = category_id
            is_crowd_by_id[instance_id] = ann.get('iscrowd', ann.get('is_crowd', False))
        
        # Prepare dataset kwargs
        dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
            "img_folder_path_in_zip": Path("./images"),
            "target_folder_path_in_zip": Path("./masks"),
            "annotations_json_path_in_zip": Path("./annotations.json"),
        }
        
        # Check if path is a ZIP file or directory
        zip_path = Path(self.path)
        has_mask_subdirs = True  # Default assumption
        
        if zip_path.suffix.lower() == '.zip':
            # Path is a ZIP file - we need to handle the internal structure
            # If ZIP contains a base folder, we need to add it to the paths
            base_folder = None
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
                
                # Check if masks are split by train/val or not
                mask_files = [f for f in all_files if '/masks/' in f and f.endswith('.png')]
                has_mask_subdirs = any('/masks/train/' in f or '/masks/val/' in f for f in mask_files)
            
            # Convert paths to use forward slashes for ZIP files
            for key in ["img_folder_path_in_zip", "target_folder_path_in_zip", "annotations_json_path_in_zip"]:
                dataset_kwargs[key] = Path(dataset_kwargs[key].as_posix())
            
            # Create the full dataset first with NO transforms (SubsetDataset will apply appropriate transforms)
            full_dataset = StrawberryDataset(
                zip_path=zip_path,
                flat_masks=(not has_mask_subdirs),
                transforms=None,  # No transforms on full dataset - SubsetDataset will apply them
                **dataset_kwargs,
            )
            
            # Calculate split indices
            total_images = len(full_dataset)
            split_idx = int(total_images * self.train_split)
            
            train_indices = list(range(0, split_idx))
            val_indices = list(range(split_idx, total_images))
            
            # Import SubsetDataset
            from datasets.subset_dataset import SubsetDataset
            
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
        else:
            # Path is a directory
            full_dataset = StrawberryDataset(
                zip_path=zip_path,
                flat_masks=False,
                transforms=None,  # No transforms on full dataset - SubsetDataset will apply them
                **dataset_kwargs,
            )
            
            # Calculate split indices
            total_images = len(full_dataset)
            split_idx = int(total_images * self.train_split)
            
            train_indices = list(range(0, split_idx))
            val_indices = list(range(split_idx, total_images))
            
            # Import SubsetDataset
            from datasets.subset_dataset import SubsetDataset
            
            # Create train and val subsets with appropriate transforms
            self.train_dataset = SubsetDataset(
                full_dataset,
                train_indices,
                transforms=self.transforms,
            )
            
            # Val dataset uses no augmentation
            val_transforms = Transforms(
                img_size=self.transforms.img_size,
                color_jitter_enabled=False,
                scale_range=(1.0, 1.0),
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


# Backward compatibility alias
COCOPanoptic = StrawberryPanoptic

