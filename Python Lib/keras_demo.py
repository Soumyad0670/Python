import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard # type: ignore
import matplotlib.pyplot as plt

class KerasDemo:
    def __init__(self):
        print("Keras version:", keras.__version__)
        
    def demonstrate_sequential_model(self):
        """Demonstrate basic Sequential model"""
        print("\n Sequential Model Example ")
        
        # Create Sequential model
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(784,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        print(model.summary())
        return model

    def demonstrate_functional_api(self):
        """Demonstrate Functional API"""
        print("\n Functional API Example ")
        
        # Create inputs
        inputs = keras.Input(shape=(784,))
        
        # Create model architecture
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name="functional_model")
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        print(model.summary())
        return model

    def create_cnn(self):
        """Create Convolutional Neural Network"""
        print("\n CNN Model Example ")
        
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        print(model.summary())
        return model

    def demonstrate_callbacks(self, model, X_train, y_train, X_val, y_val):
        """Demonstrate various callbacks"""
        print("\n Callbacks Example ")
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
        
        # TensorBoard
        tensorboard = TensorBoard(log_dir='./logs')
        
        # Train with callbacks
        history = model.fit(
            X_train, y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, checkpoint, tensorboard]
        )
        
        return history

    def demonstrate_data_augmentation(self):
        """Demonstrate image data augmentation"""
        print("\n Data Augmentation Example ")
        
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        return datagen

    def demonstrate_transfer_learning(self):
        """Demonstrate transfer learning with VGG16"""
        print("\n Transfer Learning Example ")
        
        # Load pre-trained VGG16
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(model.summary())
        return model

    def demonstrate_custom_layer(self):
        """Demonstrate creating custom layer"""
        print("\n Custom Layer Example ")
        
        class CustomDense(layers.Layer):
            def __init__(self, units):
                super(CustomDense, self).__init__()
                self.units = units
                
            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='random_normal',
                    trainable=True
                )
                self.b = self.add_weight(
                    shape=(self.units,),
                    initializer='zeros',
                    trainable=True
                )
                
            def call(self, inputs):
                return keras.backend.dot(inputs, self.w) + self.b
        
        # Use custom layer
        model = keras.Sequential([
            CustomDense(32),
            layers.Activation('relu'),
            CustomDense(10),
            layers.Activation('softmax')
        ])
        
        return model

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()

def main():
    # Create demo instance
    demo = KerasDemo()
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # Demonstrate different models
    sequential_model = demo.demonstrate_sequential_model()
    functional_model = demo.demonstrate_functional_api()
    cnn_model = demo.create_cnn()
    
    # Train model with callbacks
    history = demo.demonstrate_callbacks(
        cnn_model, 
        X_train, y_train,
        X_test, y_test
    )
    
    # Demonstrate transfer learning
    transfer_model = demo.demonstrate_transfer_learning()
    
    # Demonstrate custom layer
    custom_model = demo.demonstrate_custom_layer()
    
    # Plot results
    demo.plot_training_history(history)

if __name__ == "__main__":
    main()