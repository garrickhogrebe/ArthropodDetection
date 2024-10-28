import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from preprocess import create_normalized_dataset
import math


dataset_dir = '../output/images'

# Preprocess the images and get training/validation datasets
#train_dataset, validation_dataset, num_classes, class_labels = preprocess_images(dataset_dir)
print(tf.__version__)
train_dataset = create_normalized_dataset(
    dataset_dir          = '../data/images', 
    validation_split     = 0.2,
    subset               = "training",
    seed                 = 123,
    image_size           = (224, 224),
    batch_size           = 32
)

validation_dataset = create_normalized_dataset(
    dataset_dir          = '../data/images',
    validation_split     = 0.2,
    subset               = "validation",
    seed                 = 123,
    image_size           = (224, 224),
    batch_size           = 32
)

class_labels = train_dataset.class_names
num_classes  = len(class_labels)


model_path = "../models/pretrained/efficientnet-tensorflow2"

pretrained_model = tf.keras.models.load_model(model_path)
pretrained_model.trainable = False

model = models.Sequential([
    pretrained_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')]
)

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

batch_size = 32
train_dataset_size = sum(1 for _ in train_dataset)
steps_per_epoch = math.ceil(train_dataset_size / batch_size)

validate_dataset_size = sum(1 for _ in validation_dataset)
validation_steps =  math.ceil(validate_dataset_size / batch_size)


history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10 
)

# Evaluate model
val_loss, val_acc = model.evaluate(validation_dataset)

model.class_names = class_labels
model.save('../models/Arthropod1')

training_accuracy = history.history['sparse_categorical_accuracy']
val_accuracy = history.history['val_sparse_categorical_accuracy']

print("Training accuracy for each epoch:", training_accuracy)
print("Validation accuracy for each epoch:", val_accuracy)
print(f'Validation accuracy: {val_acc}')

