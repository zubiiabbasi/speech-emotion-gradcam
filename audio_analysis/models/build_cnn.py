"""
Build CNN Model: Create a 2D CNN for emotion classification
Input: (128, 128, 1) Mel-Spectrogram
Output: 7 Emotions (Softmax)
"""
from tensorflow import keras
from tensorflow.keras import layers

def build_emotion_cnn(input_shape=(128, 128, 1), num_classes=7):
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                      padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),  # Instead of Flatten
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def print_model_summary(model):
    """Print model architecture summary."""
    print("\n" + "="*70)
    print("CNN ARCHITECTURE")
    print("="*70)
    model.summary()
    print("="*70 + "\n")


if __name__ == "__main__":
    # Build the CNN
    print("\n" + "="*70)
    print("BUILDING EMOTION RECOGNITION CNN")
    print("="*70)
    
    model = build_emotion_cnn(input_shape=(128, 128, 1), num_classes=7)
    model = compile_model(model, learning_rate=0.001)
    
    # Print model summary
    print_model_summary(model)
    
    print("Model built successfully!")
    print(f"  Input shape: (128, 128, 1)")
    print(f"  Output: 7 emotions (Softmax)")
    print(f"  Total parameters: {model.count_params():,}")
    print()
