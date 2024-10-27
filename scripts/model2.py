import tensorflow as tf
from tensorflow.keras import layers, models
from preprocess import preprocess_images
import math


dataset_dir = '../data/images'

# Preprocess the images and get training/validation datasets
train_dataset, validation_dataset, num_classes = preprocess_images(dataset_dir)

model_path = "../models/efficientnet-tensorflow2"

pretrained_model = tf.keras.models.load_model(model_path)
pretrained_model.trainable = False

model = models.Sequential([
    pretrained_model,
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

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

model.save('..\models\10000_images')

training_accuracy = history.history['sparse_categorical_accuracy']
val_accuracy = history.history['val_sparse_categorical_accuracy']

print("Training accuracy for each epoch:", training_accuracy)
print("Validation accuracy for each epoch:", val_accuracy)
print(f'Validation accuracy: {val_acc}')

