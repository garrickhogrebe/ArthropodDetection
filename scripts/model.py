"""
model.py

This script builds a transfer learning model using a pre-trained EfficientNet model. It adds custom dense layers, trains the model on a provided dataset, and evaluates its performance. 
The model and training accuracy results are saved for later use.

Usage:
    To run this script directly, ensure the required data directory structure is present and the specified pre-trained model path is accessible. Adjust paths as needed for your environment.

Example:
    python model.py
"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from preprocess import create_normalized_dataset
import json
import os
import math

def build_and_train_model(dataset_dir, model_path, learning_rate=0.001, num_extra_layers=1, epochs=10, batch_size=32):
    """
    Builds a transfer learning model with EfficientNet as the base, adds dense layers, 
    trains it on the provided dataset, and evaluates the model's accuracy.

    Args:
        dataset_dir (str): Directory containing the dataset images for training and validation.
        model_path (str): Path to the pre-trained model to be used as the base.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Batch size for training. Default is 32.

    Returns:
        tf.keras.Model: The trained Keras model.
        dict: Training history, containing accuracy metrics for each epoch.
        float: Validation accuracy after the final epoch.
    """
    # Load training and validation datasets
    train_dataset = create_normalized_dataset(
        dataset_dir       = dataset_dir,
        validation_split  = 0.2,
        subset            = "training",
        seed              = 123,
        image_size        = (224, 224),
        batch_size        = batch_size
    )

    validation_dataset = create_normalized_dataset(
        dataset_dir       = dataset_dir,
        validation_split  = 0.2,
        subset            = "validation",
        seed              = 123,
        image_size        = (224, 224),
        batch_size        = batch_size
    )

    # Define model properties based on dataset
    class_labels = train_dataset.class_names
    num_classes  = len(class_labels)

    efficientnet_url = "https://www.kaggle.com/models/tensorflow/efficientnet/TensorFlow2/b0-feature-vector/1"
    pretrained_model = tf.keras.Sequential([
        hub.KerasLayer(efficientnet_url, trainable=False)  # Set trainable=True if you want to fine-tune
    ])

    # Build the model
    model = models.Sequential()
    model.add(pretrained_model)

    # Dynamically add Dense layers
    for _ in range(num_extra_layers):
        model.add(layers.Dense(128, activation='relu'))

    # Add the final output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer  = Adam(learning_rate=learning_rate),
                  loss       = 'sparse_categorical_crossentropy',
                  metrics    = ['sparse_categorical_accuracy'])

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data = validation_dataset,
        epochs          = epochs
    )

    # Evaluate the model on the validation set
    _, val_acc = model.evaluate(validation_dataset)

    # Store class labels in the model for reference
    model.class_names = class_labels

    return model, history.history, val_acc

if __name__ == "__main__":
    # Paths and settings
    dataset_dir       = '../data/images'
    model_path        = "../models/pretrained/efficientnet-tensorflow2"
    output_model_path = f'../models/BasicArthropodClassifier'

    # Build, train, and evaluate the model
    model, history, val_acc = build_and_train_model(dataset_dir, model_path)

    # Save the trained model
    model.save(output_model_path)
    with open(os.path.join(output_model_path, "class_names.json"), "w") as file:
        json.dump(model.class_names, file)

    # Output training and validation results
    print("Training accuracy per epoch:", history['sparse_categorical_accuracy'])
    print("Validation accuracy per epoch:", history.get('val_sparse_categorical_accuracy'))
    print(f'Final validation accuracy: {val_acc}')


