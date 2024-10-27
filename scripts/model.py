import tensorflow as tf
from preprocess import preprocess_images
from tensorflow.keras import layers, models
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

def build_model(input_shape, num_classes):
    # model = models.Sequential([
    #     layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(128, (3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Conv2D(256, (3, 3), activation='relu'),
    #     layers.MaxPooling2D((2, 2)),
    #     layers.Flatten(),
    #     layers.Dense(256, activation='relu'),
    #     layers.Dense(num_classes, activation='softmax')
    # ])

    model = tf.keras.models.load_model("models\efficientnet-tensorflow2)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    
    
    dataset_dir = 'output/images'

    # Preprocess the images and get training/validation datasets
    train_dataset, validation_dataset, num_classes = preprocess_images(dataset_dir)

    # Build the model
    model       = build_model((224, 224, 3), num_classes)

    # Train the model
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20)

    # Evaluate model
    val_loss, val_acc = model.evaluate(validation_dataset)
    
    training_accuracy = history.history['accuracy']
    print("Training accuracy for each epoch:", training_accuracy)
    print(f'Validation accuracy: {val_acc}')


    model.save('arthropod_classifier.h5')
