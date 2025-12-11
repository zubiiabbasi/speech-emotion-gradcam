"""
ASR Script: Extract transcripts from TESS audio files using Whisper
"""
import whisper
from pathlib import Path

def extract_transcript(audio_path, model_name="base"):

    try:
        # Load the Whisper model
        model = whisper.load_model(model_name)
        
        # Transcribe audio
        result = model.transcribe(audio_path)
        
        # Extract and return the text
        transcript = result["text"].strip()
        return transcript
    
    except Exception as e:
        print(f"Error transcribing {audio_path}: {str(e)}")
        return ""


def process_tess_directory(data_dir="data", model_name="base"):

    results = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Data directory not found: {data_dir}")
        return results
    
    # Get all emotion folders
    emotion_folders = sorted([f for f in data_path.iterdir() if f.is_dir()])
    
    for emotion_folder in emotion_folders:
        emotion_folder_name = emotion_folder.name
        print(f"\nProcessing emotion: {emotion_folder_name}")
        
        # Extract speaker type (OAF or YAF) from folder name
        speaker = "OAF" if emotion_folder_name.startswith("OAF") else ("YAF" if emotion_folder_name.startswith("YAF") else "Unknown")
        # Extract emotion label (remove speaker prefix)
        emotion = emotion_folder_name.replace("OAF_", "").replace("YAF_", "")
        
        # Get all audio files in the emotion folder
        audio_files = sorted(emotion_folder.glob("*.wav"))
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"  [{i}/{len(audio_files)}] Transcribing: {audio_file.name}")
            
            # Extract transcript
            transcript = extract_transcript(str(audio_file), model_name)
            
            # Store result
            results.append({
                "path": str(audio_file),
                "speaker": speaker,
                "emotion": emotion,
                "transcript": transcript
            })
    
    return results


if __name__ == "__main__":
    # Process all TESS files
    print("Starting ASR extraction from TESS dataset...")
    results = process_tess_directory(model_name="base")
    print(f"\n\nTotal files processed: {len(results)}")
