"""
Build CNN Model: Create a 2D CNN for emotion classification
Input: (128, 128, 1) Mel-Spectrogram
Output: 7 Emotions (Softmax)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

l2 = keras.regularizers.l2(0.0005)


class SparseCategoricalCrossentropyWithLabelSmoothing(keras.losses.Loss):
    """Sparse integer labels + label smoothing (works on Keras builds without SCC(..., label_smoothing=))."""

    def __init__(self, label_smoothing=0.1, name="sparse_ce_label_smooth", **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = float(label_smoothing)

    def get_config(self):
        cfg = super().get_config()
        cfg["label_smoothing"] = self.label_smoothing
        return cfg

    def call(self, y_true, y_pred):
        ls = tf.cast(self.label_smoothing, y_pred.dtype)
        n_classes = tf.cast(tf.shape(y_pred)[-1], y_pred.dtype)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        depth = tf.shape(y_pred)[-1]
        one_hot = tf.one_hot(y_true, depth=depth, dtype=y_pred.dtype)
        soft = one_hot * (1.0 - ls) + ls / n_classes
        return tf.reduce_mean(
            keras.losses.categorical_crossentropy(soft, y_pred, from_logits=False)
        )


def build_emotion_cnn(input_shape=(128, 128, 1), num_classes=7):
    
    model = keras.Sequential([
        # Block 1 (16 filters — smaller capacity for ~1.4k samples)
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu',
                      padding='same', input_shape=input_shape,
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',
                      kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.0003, label_smoothing=0.1):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = SparseCategoricalCrossentropyWithLabelSmoothing(label_smoothing=label_smoothing)

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

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
    model = compile_model(model, learning_rate=0.0003)
    
    # Print model summary
    print_model_summary(model)
    
    print("Model built successfully!")
    print(f"  Input shape: (128, 128, 1)")
    print(f"  Output: 7 emotions (Softmax)")
    print(f"  Total parameters: {model.count_params():,}")
    print()
