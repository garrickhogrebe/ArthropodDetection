# Arthropod Detection and Classification

## Project Overview
This project implements a deep learning model using TensorFlow to classify various arthropod species. The dataset is sourced from the iNaturalist API and preprocessed to train an EfficientNet-based model. The project includes utilities for data collection, preprocessing, training, and evaluating the model, along with sample visualizations of predictions.

## Features
- **Automated Data Collection**: Downloads images and metadata for specific arthropod species from iNaturalist.
- **Data Preprocessing**: Normalizes and splits data into training and validation sets.
- **Model Training and Evaluation**: Trains an EfficientNet model and evaluates performance using metrics such as accuracy, confusion matrix, and classification report.
- **Visualization Tools**: Includes utilities to visualize sample predictions with overlayed class labels and confidence scores.

## Project Structure

- **`inat_api.py`**: Script to download images and metadata for specified taxa from the iNaturalist API.
- **`preprocess.py`**: Contains functions to create normalized datasets for training and validation.
- **`model.py`** & **`model2.py`**: Scripts for building, training, and saving the model using TensorFlow.
- **`verify.py`**: Script to verify TensorFlow installation and GPU access.
- **`jupyterUtils.py`**: Utility functions and classes for Jupyter notebooks, including a progress bar and image prediction functions.
- **`ModelEvaluater.ipynb`**: Jupyter Notebook for evaluating the model with metrics and sample visualizations.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/ArthropodDetection.git
   ```
   ```bash
   cd ArthropodDetection
   ```

2. **Install Dependencies**:
   Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify TensorFlow and GPU** (Optional):
   Run `verify.py` to confirm that TensorFlow is installed and can access the GPU:
   ```bash
   cd scripts
   ```
   ```bash
   python verify.py
   ```

## Usage

### 1. Data Collection
Use `inat_api.py` to download images and metadata for the chosen taxa (species groups). Adjust parameters such as `num_images` inside the script main as needed:
```bash
python inat_api.py
```

### 2. Train the Model
Train the model using either `model.py`:
```bash
python model2.py
```

### 3. Evaluate and Visualize
Open `ModelEvaluater.ipynb` in Jupyter Notebook to run detailed evaluations, including confusion matrix and classification report, and to visualize sample predictions with overlays.

## Example Notebook
`ModelEvaluater.ipynb` contains step-by-step instructions for evaluating the model, generating metrics, and visualizing predictions. Open this notebook in Jupyter to explore model performance in detail.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **TensorFlow** for model training, pretrained models, and evaluation.
- **iNaturalist** for the dataset API used for training data collection.

