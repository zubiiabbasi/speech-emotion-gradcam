"""
CSV Generator: Create tess_metadata.csv with Path, Emotion, and Transcript columns
"""
import csv
from pathlib import Path
from extract_transcript import process_tess_directory

def generate_metadata_csv(data_dir=None, output_file="tess_metadata.csv", model_name="base"):
    
    # If no data_dir provided, use project root/data
    if data_dir is None:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
    else:
        data_dir = Path(data_dir)

    print(f"Generating metadata CSV from {data_dir}...")
    
    # Extract transcripts from all TESS files
    results = process_tess_directory(str(data_dir), model_name)
    
    if not results:
        print("No files found to process!")
        return
    
    # Write to CSV
    output_path = Path(output_file)
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Path', 'Speaker', 'Emotion', 'Transcript']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data rows
            for result in results:
                writer.writerow({
                    'Path': result['path'],
                    'Speaker': result['speaker'],
                    'Emotion': result['emotion'],
                    'Transcript': result['transcript']
                })
        
        print(f"\nCSV file created successfully: {output_path}")
        print(f"  Total records: {len(results)}")
        
        # Display summary statistics
        emotion_counts = {}
        for result in results:
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count} files")
        
        # Display speaker distribution
        speaker_counts = {}
        for result in results:
            speaker = result['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        print("\nSpeaker distribution:")
        for speaker, count in sorted(speaker_counts.items()):
            print(f"  {speaker}: {count} files")
    
    except Exception as e:
        print(f"Error writing CSV file: {str(e)}")


if __name__ == "__main__":
    generate_metadata_csv()
