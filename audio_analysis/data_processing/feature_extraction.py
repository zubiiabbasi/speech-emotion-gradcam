"""
Feature Extraction: Load audio files, convert to Mel-Spectrograms, and pad to 128x128
"""
import librosa
import numpy as np
from pathlib import Path
import pickle
from sklearn.preprocessing import LabelEncoder

def extract_mel_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512, max_duration=5):

    try:
        # Load audio with a maximum duration
        y, sr = librosa.load(audio_path, sr=None, duration=max_duration)
        
        # Extract Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize/Pad to (128, 128)
        if mel_spec_db.shape[1] < 128:
            # Pad with zeros if too short
            mel_spec_db = np.pad(
                mel_spec_db, 
                ((0, 0), (0, 128 - mel_spec_db.shape[1])), 
                mode='constant', 
                constant_values=-80
            )
        else:
            # Crop if too long
            mel_spec_db = mel_spec_db[:, :128]
        
        return mel_spec_db
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


def process_tess_dataset(data_dir="data", output_file="tess_features.pkl"):

    data_path = Path(data_dir)
    output_path = Path(output_file)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return None
    
    features = []
    labels = []
    emotion_list = []
    
    # Map folder names to emotion labels (normalize case and spacing)
    # Handles variations: Fear/fear, Sad/sad, Pleasant_surprise/pleasant_surprised
    emotion_mapping = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'pleasant_surprised': 'pleasant_surprise',
        'pleasant_surprise': 'pleasant_surprise'
    }
    
    # Get all emotion folders
    emotion_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
    
    print(f"Found {len(emotion_folders)} emotion folders\n")
    
    for emotion_folder in emotion_folders:
        folder_name = emotion_folder.name
        
        # Extract emotion label (remove speaker prefix like OAF_ or YAF_)
        emotion_label = folder_name.replace("OAF_", "").replace("YAF_", "").lower()
        
        # Map to standardized emotion label
        emotion = emotion_mapping.get(emotion_label, emotion_label)
        
        if emotion not in emotion_list:
            emotion_list.append(emotion)
        
        # Get all audio files in the emotion folder
        audio_files = sorted(emotion_folder.glob("*.wav"))
        
        print(f"Processing emotion: {folder_name} ({emotion}) - {len(audio_files)} files")
        
        for i, audio_file in enumerate(audio_files, 1):
            # Extract Mel-Spectrogram
            mel_spec = extract_mel_spectrogram(str(audio_file))
            
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(emotion)
                
                if i % 50 == 0:
                    print(f"  Processed {i}/{len(audio_files)} files...")
        
        print(f"Completed {folder_name}\n")
    
    # Convert to numpy arrays
    features = np.array(features)
    
    # Encode emotion labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    features = np.expand_dims(features, axis=-1)
    
    # Save features and labels
    data = {
        'features': features,
        'labels': encoded_labels,
        'label_encoder': label_encoder,
        'emotion_list': sorted(emotion_list)
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Features saved to: {output_path}\n")
    
    return data


if __name__ == "__main__":
    from pathlib import Path
    # Find data directory
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir = Path("..") / "data"
    
    output_file = Path("tess_features.pkl")
    if not output_file.parent.exists():
        output_file = Path(".") / "tess_features.pkl"
    
    data = process_tess_dataset(
        data_dir=str(data_dir), 
        output_file=str(output_file)
    )
