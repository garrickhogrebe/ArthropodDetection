"""
jupyterUtils.py

This module provides utility classes and functions designed for Jupyter Notebook environments, including:
- A progress bar class to display a rotating progress animation.
- Functions to evaluate a TensorFlow model on a validation dataset.
- A function to predict the class of an image and overlay the prediction and confidence score on the image.

Usage:
    Import this module into a Jupyter Notebook or Python script to use the ProgressBar class, 
    evaluate a model, or make predictions with overlays.

Example:
    from jupyterUtils import ProgressBar, evaluate_dataset, predict_image
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from preprocess import create_normalized_dataset
import threading
import time
import sys


class ProgressBar:
    """
    Displays a rotating progress animation on a separate line in the console.
    
    Args:
        message (str): The message to display alongside the animation.
        interval (float): Time between animation updates in seconds.
    
    Usage:
        progress_bar = ProgressBar("Loading data")
        # Perform some task
        progress_bar.stop()
    """
    def __init__(self, message="Processing", interval=0.4):
        self.interval    = interval
        self.message     = message
        self._stop_event = threading.Event()
        self.thread      = threading.Thread(target=self._progress_animation)
        self.thread.start()

    def _progress_animation(self):
        """Handles the progress animation on a separate line to avoid overlap with other output."""
        print("")  # Add an extra line for the loading bar
        while not self._stop_event.is_set():
            for char in "|/-\\":
                sys.stdout.write(f"\033[F\r{self.message} {char}")
                sys.stdout.flush()
                time.sleep(self.interval)
                if self._stop_event.is_set():
                    break
        sys.stdout.write("\033[F\r" + " " * (len(self.message) + 10) + "\r")  
        sys.stdout.flush()

    def stop(self):
        """Stops the progress bar and clears the line."""
        self._stop_event.set()
        self.thread.join()


def evaluate_dataset(model, validation_dataset):
    """
    Evaluates a model on a validation dataset and returns arrays of true and predicted labels.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to evaluate.
        validation_dataset (tf.data.Dataset): The validation dataset to evaluate the model on.

    Returns:
        tuple: Two numpy arrays - true labels and predicted labels for the dataset.
    """
    true = []
    pred = []

    # Iterate through the dataset and collect predictions
    for images, labels in validation_dataset:
        predictions       = model.predict(images, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true.extend(labels.numpy())
        pred.extend(predicted_classes)

    true = np.array(true)
    pred = np.array(pred)

    return true, pred


def predict_image(model, img_path, class_names, font_size=80):
    """
    Loads an image, preprocesses it for the model, and overlays the predicted class and confidence on the image.
    
    Args:
        model (tf.keras.Model): The TensorFlow model used for prediction.
        img_path (str): Path to the image file.
        class_names (list): List of class names corresponding to model output indices.
        font_size (int): Font size for overlay text. Default is 80.

    Displays:
        The image with overlayed predicted class and confidence score.
    """
    # Load and preprocess the image
    img           = Image.open(img_path)
    img_resized   = img.resize((224, 224))  # Adjust for model input size
    img_array     = image.img_to_array(img_resized)
    img_array     = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict with the model
    predictions           = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence            = predictions[0][predicted_class_index]
    predicted_class       = class_names[predicted_class_index]

    # Overlay text on the image
    draw         = ImageDraw.Draw(img)
    overlay_text = f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}"

    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Load Arial font if available
    except IOError:
        font = ImageFont.load_default()  # Fallback to default font if Arial is unavailable

    # Define text position and styling
    text_position = (10, 10)
    text_bg_color = (0, 0, 0)  # Black background for text
    text_color    = (255, 255, 255)  # White text color
    text_size     = draw.textsize(overlay_text, font=font)

    # Draw the text background and overlay
    draw.rectangle([text_position, (text_position[0] + text_size[0], text_position[1] + text_size[1])],
                   fill=text_bg_color)
    draw.text(text_position, overlay_text, fill=text_color, font=font)

    # Display the image with overlay using Matplotlib
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    
    # Print prediction to console
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
