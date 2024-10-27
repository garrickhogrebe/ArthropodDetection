import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def preprocess_images(dataset_dir, img_size=(224, 224), batch_size=32, validation_split=0.2):
    
    # Load the training dataset
    train_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=img_size,  # Resize all images to the target size
        batch_size=batch_size
    )

    # Load the validation dataset
    validation_dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    num_classes = len(train_dataset.class_names)
    print("Class names:", train_dataset.class_names)

    # Normalize pixel values to [0, 1] range
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    # Apply normalization to the datasets
    train_dataset      = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, validation_dataset, num_classes

if __name__ == "__main__":
    dataset_dir = 'data'

    # Preprocess the images and get training/validation datasets
    train_dataset, validation_dataset = preprocess_images(dataset_dir)

    # Output the class names and some dataset information
    
    print("Training dataset:", train_dataset)
    print("Validation dataset:", validation_dataset)