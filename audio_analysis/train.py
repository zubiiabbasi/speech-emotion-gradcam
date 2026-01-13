"""
Train CNN Model: Load features, train, and save best model
Trains for 30-50 epochs with 80/20 train-test split
Saves best model as best_model.h5
"""
import pickle
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Import model building functions
from models import build_emotion_cnn, compile_model

def load_features(features_file="data_processing/tess_features.pkl"):

    # Find features file (works from audio_analysis/or project root)
    possible_paths = [
        Path(features_file),
        Path("audio_analysis") / features_file,
    ]
    
    features_path = None
    for path in possible_paths:
        if path.exists():
            features_path = path
            print(f"Found features file: {features_path}\n")
            break
    
    if features_path is None:
        print("ERROR: Could not find features pickle file!")
        print("Tried these paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nFirst run: python audio_analysis/data_processing/feature_extraction.py")
        return None, None, None, None
    
    try:
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data['features']
        labels = data['labels']
        label_encoder = data['label_encoder']
        emotion_list = data['emotion_list']
        
        print(f"Features loaded successfully!")
        print(f"  Shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Emotions: {emotion_list}\n")
        
        return features, labels, label_encoder, emotion_list
    
    except Exception as e:
        print(f"Error loading features: {str(e)}")
        return None, None, None, None


def prepare_data(features, labels, test_size=0.3, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels  # Ensures balanced split
    )
    
    print(f"Data Split (70/30):")
    print(f"  Train samples: {X_train.shape[0]} (70%)")
    print(f"  Test samples: {X_test.shape[0]} (30%)")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape: {X_test.shape}\n")
    
    return X_train, X_test, y_train, y_test


def create_callbacks(model_name="best_model.h5"):

    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            filepath=model_name,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Stop early if validation accuracy stops improving
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Stop if no improvement for 10 epochs
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate if validation accuracy plateaus
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def train_model(model, X_train, X_test, y_train, y_test, epochs=50, batch_size=32):

    print(f"\n{'='*70}")
    print(f"TRAINING EMOTION RECOGNITION CNN")
    print(f"{'='*70}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total training samples: {X_train.shape[0]}")
    print(f"{'='*70}\n")
    
    callbacks = create_callbacks(model_name="best_model.h5")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    """Main training pipeline - Train and save model only."""
    
    print(f"\n{'='*70}")
    print(f"EMOTION RECOGNITION CNN - TRAINING")
    print(f"{'='*70}\n")
    
    # 1. Load features
    features, labels, label_encoder, emotion_list = load_features()
    if features is None:
        return
    
    # 2. Prepare data (70/30 split)
    X_train, X_test, y_train, y_test = prepare_data(features, labels)
    
    # 3. Build model
    print(f"Building CNN model...")
    model = build_emotion_cnn(input_shape=(128, 128, 1), num_classes=len(emotion_list))
    model = compile_model(model, learning_rate=0.001)
    print(f"Model built with {model.count_params():,} parameters\n")
    
    # 4. Train model
    history = train_model(
        model, X_train, X_test, y_train, y_test,
        epochs=20, 
        batch_size=32
    )


if __name__ == "__main__":
    main()
