import os
import shutil
import random
from pathlib import Path
from typing import Union, Optional


def build_dataset(
    source_dir: str = "dataset",
    output_dir: str = "training_data",
    train_split: Union[float, int] = 0.7,
    val_split: Union[float, int] = 0.15,
    test_split: Union[float, int] = 0.15,
    seed: int = 42,
    clear_output: bool = True,
    balance_classes: bool = False,
    zip_output: bool = False,
    zip_path: Optional[str] = None
):
    """
    Build a training dataset from marked/unmarked images.
    
    Handles imbalanced datasets by processing each class independently.
    
    Args:
        source_dir: Directory containing 'marked' and 'unmarked' subdirectories
        output_dir: Output directory for train/val/test splits
        train_split: Percentage (0-1) or number of images for training set
        val_split: Percentage (0-1) or number of images for validation set
        test_split: Percentage (0-1) or number of images for test set
        seed: Random seed for reproducibility
        clear_output: Whether to clear the output directory before building
        balance_classes: If True, uses the same number of images from each class
                (limited by the smallest class size)
        zip_output: If True, compresses output_dir into a .zip file
        zip_path: Optional output path for the .zip file. If None, uses
              "{output_dir}.zip" in the same parent directory.
    """
    random.seed(seed)
    
    # Setup paths
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Categories to process
    categories = ["marked", "unmarked"]
    
    # Validate source directory
    for category in categories:
        category_path = source_path / category
        if not category_path.exists():
            raise ValueError(f"Source directory not found: {category_path}")
    
    # Clear or create output directory
    if clear_output and output_path.exists():
        shutil.rmtree(output_path)
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        for category in categories:
            (output_path / split / category).mkdir(parents=True, exist_ok=True)
    
    # First pass: collect all images and check for class imbalance
    category_images = {}
    for category in categories:
        category_path = source_path / category
        image_files = [
            f for f in category_path.iterdir() 
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        ]
        category_images[category] = image_files
    
    # Check for class imbalance
    class_counts = {cat: len(imgs) for cat, imgs in category_images.items()}
    print("\n" + "="*50)
    print("Original Dataset Statistics:")
    print("="*50)
    for category, count in class_counts.items():
        print(f"{category}: {count} images")
    
    min_class_size = min(class_counts.values())
    max_class_size = max(class_counts.values())
    
    if min_class_size != max_class_size:
        imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
        print(f"\nClass imbalance detected: {imbalance_ratio:.2f}x difference")
        if balance_classes:
            print(f"Balancing classes: using {min_class_size} images from each class")
    
    # Process each category
    for category in categories:
        print(f"\nProcessing {category} images...")
        
        # Get image files from first pass
        image_files = category_images[category].copy()
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"Warning: No images found in {category}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Apply class balancing if requested
        if balance_classes and total_images > min_class_size:
            image_files = image_files[:min_class_size]
            total_images = min_class_size
            print(f"Balanced to {total_images} images")
        else:
            print(f"Using {total_images} images")
        
        # Calculate split sizes
        if isinstance(train_split, float):
            # Percentage-based splits
            train_size = int(total_images * train_split)
            val_size = int(total_images * val_split)
            test_size = total_images - train_size - val_size
        else:
            # Number-based splits
            train_size = min(train_split, total_images)
            val_size = min(val_split, total_images - train_size)
            test_size = min(test_split, total_images - train_size - val_size)
        
        print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Split and copy files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:train_size + val_size + test_size]
        
        # Copy to respective directories
        for img_file in train_files:
            shutil.copy2(img_file, output_path / "train" / category / img_file.name)
        
        for img_file in val_files:
            shutil.copy2(img_file, output_path / "val" / category / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, output_path / "test" / category / img_file.name)
        
        print(f"Copied {len(train_files)} to train, {len(val_files)} to val, {len(test_files)} to test")
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset build complete!")
    print("="*50)
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        for category in categories:
            count = len(list((output_path / split / category).iterdir()))
            print(f"  {category}: {count} images")

    # Optional: compress output directory into a zip file
    if zip_output:
        if zip_path:
            zip_target = Path(zip_path)
            zip_target.parent.mkdir(parents=True, exist_ok=True)
            zip_base = zip_target.with_suffix("")
        else:
            zip_base = output_path

        archive_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_path)
        print(f"\nCreated zip archive: {archive_path}")


if __name__ == "__main__":
    # Example 1: Using percentages (70% train, 15% val, 15% test)
    build_dataset(
        source_dir="pencil_dataset",
        output_dir="training6",
        train_split=0.85,
        val_split=0,
        test_split=0.15,
        seed=42,
        balance_classes=True,
        zip_output=True
    )
    
    # Example 2: Using fixed numbers (100 train, 20 val, 20 test)
    # build_dataset(
    #     source_dir="dataset",
    #     output_dir="training_data",
    #     train_split=100,
    #     val_split=20,
    #     test_split=20,
    #     seed=42
    # )
