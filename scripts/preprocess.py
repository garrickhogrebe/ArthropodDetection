import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def create_normalized_dataset(dataset_dir, validation_split, subset, seed, image_size, batch_size, shuffle=True):

    # Load data set from directory
    dataset = image_dataset_from_directory(
        dataset_dir,
        validation_split    = validation_split,
        subset              = subset,
        seed                = seed,
        image_size          = image_size, #TODO: preserve aspect ratio and padd
        batch_size          = batch_size,
        shuffle             = shuffle
    )

    # Normalize pixel values to [0, 1] range
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_dataset  = dataset.map(lambda x, y: (normalization_layer(x), y))

    # Copy over meta data
    normalized_dataset.class_names  = dataset.class_names

    return normalized_dataset



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

    class_labels = train_dataset.class_names

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

    return train_dataset, validation_dataset, num_classes, class_labels

if __name__ == "__main__":
    dataset_dir = '../data/images'

    # Preprocess the images and get training/validation datasets
    train_dataset, validation_dataset, num_classes, class_labels = preprocess_images(dataset_dir)

    # Output the class names and some dataset information
    
    print("Training dataset:", train_dataset)
    print("Validation dataset:", validation_dataset)
    print("Class labels:", class_labels)