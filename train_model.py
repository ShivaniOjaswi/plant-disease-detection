import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from model import DISEASE_CLASSES, IMG_SIZE, NUM_CLASSES

def create_model():
    """Create a CNN model using MobileNetV2 as base"""
    # Create the base model from MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(dataset_path, epochs=10, batch_size=32):
    """Train the model on the dataset"""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory {dataset_path} not found.")
        print("Please download the PlantVillage dataset and try again.")
        return None
    
    # Create model directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load and prepare the data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create the model
    model = create_model()
    
    # Model checkpoint to save the best model
    checkpoint_path = "models/checkpoint_model.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    print("Training the model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save the final model
    model.save("models/plant_disease_model.h5")
    print("Model saved to models/plant_disease_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    
    return model

def main():
    print("Plant Disease Detection Model Training")
    print("======================================")
    
    # Ask for dataset path
    default_dataset_path = "dataset/PlantVillage"
    dataset_path = input(f"Enter the path to your dataset [default: {default_dataset_path}]: ") or default_dataset_path
    
    # Ask for number of epochs
    try:
        epochs = int(input("Enter the number of training epochs [default: 10]: ") or "10")
    except ValueError:
        print("Invalid input. Using default: 10 epochs.")
        epochs = 10
    
    # Ask for batch size
    try:
        batch_size = int(input("Enter the batch size [default: 32]: ") or "32")
    except ValueError:
        print("Invalid input. Using default: batch size 32.")
        batch_size = 32
    
    # Train the model
    train_model(dataset_path, epochs, batch_size)
    
    print("\nTraining complete! You can now use the model for predictions.")

if __name__ == "__main__":
    main() 