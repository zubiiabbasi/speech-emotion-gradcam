"""
CSV Generator: Create tess_metadata.csv with Path, Emotion, and Transcript columns
"""
import csv
from pathlib import Path
from extract_transcript import process_tess_directory

def generate_metadata_csv(data_dir, output_file, model_name="base"):
    
    data_dir = Path(data_dir)
    output_file = Path(output_file)

    print(f"Generating metadata CSV from {data_dir}...")
    
    # Extract transcripts from all TESS files
    results = process_tess_directory(str(data_dir), model_name)
    
    if not results:
        print("No files found to process!")
        return
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
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
        
        print(f"\nCSV file created successfully: {output_file}")
        print(f"  Total records: {len(results)}")
    
    except Exception as e:
        print(f"Error writing CSV file: {str(e)}")


if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    text_analysis_dir = Path(__file__).parent
    output_file = text_analysis_dir / "tess_metadata.csv"
    
    # Generate CSV
    generate_metadata_csv(str(data_dir), str(output_file))
