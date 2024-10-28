"""
preprocess.py

This script contains functions for creating normalized datasets for training and validation.
The `create_normalized_dataset` function loads images from a directory, normalizes pixel values, and 
organizes images into a format suitable for TensorFlow training. Run this script directly to create 
and view sample datasets.

Usage:
    Adjust `dataset_dir` in the `__main__` block to point to the image directory and then run:

    python preprocess.py
"""

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def create_normalized_dataset(dataset_dir, validation_split, subset, seed, image_size, batch_size, shuffle=True):
    """
    Creates a normalized dataset for either training or validation by loading images, resizing, 
    and normalizing pixel values to the [0, 1] range.

    Args:
        dataset_dir (str): Directory path where the dataset images are stored.
        validation_split (float): Fraction of data to use for validation.
        subset (str): Either 'training' or 'validation' to specify dataset type.
        seed (int): Seed for random operations for reproducibility.
        image_size (tuple): Target size for resizing images.
        batch_size (int): Number of images per batch.
        shuffle (bool): Whether to shuffle the dataset. Default is True.

    Returns:
        tf.data.Dataset: A normalized TensorFlow dataset ready for training or evaluation, 
                         with class names accessible via dataset.class_names.
    """
    # Load data set from directory
    dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split    = validation_split,
        subset              = subset,
        seed                = seed,
        image_size          = image_size,  # TODO: preserve aspect ratio and pad
        batch_size          = batch_size,
        shuffle             = shuffle
    )

    # Normalize pixel values to [0, 1] range
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset  = dataset.map(lambda x, y: (normalization_layer(x), y))

    # Copy over metadata for class names
    normalized_dataset.class_names = dataset.class_names

    return normalized_dataset

if __name__ == "__main__":
    dataset_dir        = '../data/images'
    validation_split   = 0.2
    image_size         = (224, 224)
    batch_size         = 32
    seed               = 123

    # Create training and validation datasets
    train_dataset      = create_normalized_dataset(dataset_dir, validation_split, "training", seed, image_size, batch_size)
    validation_dataset = create_normalized_dataset(dataset_dir, validation_split, "validation", seed, image_size, batch_size)

    # Output sample information about datasets
    print("Training dataset:", train_dataset)
    print("Validation dataset:", validation_dataset)
    print("Class labels:", train_dataset.class_names)
