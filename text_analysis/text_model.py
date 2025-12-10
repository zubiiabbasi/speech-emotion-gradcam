"""
Text Model: Train a Naive Bayes classifier on TESS transcripts to show low accuracy
"""

import csv
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


def load_metadata_csv(csv_file="tess_metadata.csv"):
    """
    Load the metadata CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return None
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records from {csv_file}")
    return df


def train_text_model(csv_file="tess_metadata.csv", test_size=0.2):
    """
    Train a Naive Bayes classifier on text transcripts.
    
    Args:
        csv_file (str): Path to the metadata CSV
        test_size (float): Proportion of data for testing
    """
    print("=" * 60)
    print("TEXT-ONLY EMOTION CLASSIFIER")
    print("=" * 60)
    
    # Load data
    df = load_metadata_csv(csv_file)
    if df is None or len(df) == 0:
        print("No data to process!")
        return
    
    # Check for empty transcripts
    df = df[df['Transcript'].notna() & (df['Transcript'].str.len() > 0)]
    
    if len(df) == 0:
        print("No valid transcripts found!")
        return
    
    print(f"Valid records with transcripts: {len(df)}\n")
    
    # Prepare data
    X = df['Transcript'].values
    y = df['Emotion'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}\n")
    
    # Vectorize text using TF-IDF
    print("Vectorizing transcripts using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Feature dimensions: {X_train_vec.shape[1]}\n")
    
    # Train Naive Bayes model
    print("Training Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Get baseline accuracy (random guessing)
    unique_emotions = len(set(y))
    baseline_accuracy = 1.0 / unique_emotions
    print(f"Baseline Accuracy (random guessing): {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"\n⚠️  Note: Accuracy is only ~{accuracy*100:.0f}%, which is {'above' if accuracy > baseline_accuracy else 'below'} baseline!")
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred))
    
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    emotions = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=emotions)
    
    # Format confusion matrix display
    print("\nEmotions:", emotions)
    print(cm)
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
The text-only classifier achieves only {accuracy*100:.2f}% accuracy.
This demonstrates that TRANSCRIPTS ALONE are insufficient for 
emotion recognition in speech. Acoustic features (prosody, pitch, 
energy, etc.) are essential for better performance.

This justifies the use of Grad-CAM visualization on the audio features!
""")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred
    }


if __name__ == "__main__":
    train_text_model()
