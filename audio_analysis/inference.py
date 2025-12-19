"""
Inference Pipeline for Speech Emotion Recognition

"""

import numpy as np
import librosa
import tensorflow as tf
from pathlib import Path
from gtts import gTTS
import warnings

warnings.filterwarnings('ignore')


def load_model(model_path="best_model.h5"):
   
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_audio(audio_path, n_mels=128, n_fft=2048, hop_length=512, max_duration=5):
    
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    
    try:
        # Load audio with ORIGINAL sample rate (matching training)
        y, sr = librosa.load(str(audio_path), sr=None, duration=max_duration)
        
        # Extract Mel-Spectrogram (exact parameters from training)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # Convert to dB scale (matching training)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # DO NOT normalize - model was trained on raw dB values
        
        # Pad or crop to (128, 128) with -80 constant (matching training)
        if mel_spec_db.shape[1] < 128:
            mel_spec_db = np.pad(
                mel_spec_db, 
                ((0, 0), (0, 128 - mel_spec_db.shape[1])), 
                mode='constant', 
                constant_values=-80
            )
        else:
            mel_spec_db = mel_spec_db[:, :128]
        
        # Add channel dimension
        mel_spec_final = np.expand_dims(mel_spec_db, axis=-1).astype(np.float32)
        
        return mel_spec_final
    
    except Exception as e:
        raise ValueError(f"Error processing audio file: {e}")


def predict_emotion(audio_path, model, label_encoder, return_confidence=False):
    
    # Preprocess audio
    mel_spec = preprocess_audio(audio_path)
    
    # Prepare batch
    mel_spec_batch = np.expand_dims(mel_spec, axis=0)
    
    # Make prediction
    predictions = model.predict(mel_spec_batch, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(predictions[0][pred_idx])
    
    if return_confidence:
        confidence_dict = {
            label_encoder.classes_[i]: float(predictions[0][i])
            for i in range(len(label_encoder.classes_))
        }
        return pred_label, confidence_dict
    
    return pred_label


def generate_tts_response(emotion_label, language='en', use_label_only=True, transcript=None, slow=False):
    
    # Determine text to speak
    if use_label_only:
        text_to_speak = f"This person seems {emotion_label}."
    else:
        if transcript is None:
            raise ValueError("transcript is required when use_label_only=False")
        text_to_speak = f"The emotion detected is {emotion_label}. The transcript says: {transcript}"
    
    try:
        # Create TTS object
        tts = gTTS(text=text_to_speak, lang=language, slow=slow)
        
        # Save to file
        output_path = Path("audio_analysis/tts_output.mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tts.save(str(output_path))
        
        return str(output_path)
    
    except Exception as e:
        raise ValueError(f"Error generating TTS: {e}")


# =====================================================
# Utility: Load label encoder from pickled data
# =====================================================
def load_label_encoder(features_pkl_path="audio_analysis/data_processing/tess_features.pkl"):
   
    import pickle
    
    features_pkl_path = Path(features_pkl_path)
    
    if not features_pkl_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_pkl_path}")
    
    with open(features_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['label_encoder']


# =====================================================
# Grad-CAM: Generate visual explanations for predictions
# =====================================================
def compute_gradcam(model, input_data, class_idx):
    
    # Find the last convolutional layer
    last_conv_layer_name = None
    for layer in model.layers[::-1]:
        if 'conv' in layer.name:
            last_conv_layer_name = layer.name
            break
    
    if last_conv_layer_name is None:
        print("No convolutional layer found in model")
        return None
    
    print(f"[GradCAM] Using layer: {last_conv_layer_name}")
    
    # Create a model for the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )
    
    # Make layers trainable for gradient computation
    for layer in grad_model.layers:
        layer.trainable = True
    
    # Convert to tensor
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        conv_output, model_output = grad_model(input_tensor)
        class_channel = model_output[:, class_idx]
    
    # Get gradients
    grads = tape.gradient(class_channel, conv_output)
    
    if grads is None:
        print("[GradCAM] Warning: Could not compute gradients, using fallback")
        # Fallback: use channel importance without gradients
        grads = tf.ones_like(conv_output)
    
    print(f"[GradCAM] Gradients computed. Shape: {grads.shape}")
    
    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    
    # Create heatmap by weighting feature maps with pooled gradients
    heatmap = tf.zeros((conv_output.shape[0], conv_output.shape[1]))
    for i in tf.range(pooled_grads.shape[0]):
        heatmap = heatmap + pooled_grads[i] * conv_output[:, :, i]
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)  # ReLU activation
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()


if __name__ == "__main__":
    print("This module is intended to be imported and used in other scripts.")