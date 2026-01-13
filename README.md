# speech-emotion-gradcam

Speech emotion recognition on the TESS dataset using mel-spectrograms + a CNN, with explainability via Grad-CAM and an inference demo that can also speak the predicted emotion (TTS).

This repo contains two tracks:

- **Audio emotion recognition** (TensorFlow/Keras CNN + Grad-CAM)
- **Text emotion analysis** (Whisper ASR + classic ML in `text_analysis/`)

---

## What we built

### Audio pipeline (end-to-end)

1. **Feature extraction**: converts each `.wav` to a 128×128 mel-spectrogram (dB) and saves a pickle dataset.
2. **Training**: trains a CNN and saves the best model checkpoint to `audio_analysis/best_model.h5`.
3. **Evaluation notebook**: reports accuracy, classification report, confusion matrix, and per-class accuracy.
4. **Inference demo notebook**:
	 - predicts emotion for a chosen audio file
	 - plays TTS saying the predicted emotion
	 - visualizes Grad-CAM heatmaps on the spectrogram

### Explainability

Grad-CAM is implemented for the CNN to highlight time–frequency regions that most influenced the predicted class.

---

## Key finding (why you can see 100% accuracy)

On TESS, **very high / even 100% accuracy** can happen with a random per-sample split (like 70/30), even if your code is correct.

Main reasons:

- **Only two speakers** (OAF, YAF) → the model can learn speaker-specific cues.
- **Repeated phrases across the dataset** → the model can partially memorize phrase/recording patterns.
- A random split mixes speakers/phrases across train and test, so test data is not truly “unseen” in the way we care about for generalization.

So: **perfect accuracy here is a sign the evaluation is not strict enough**, not necessarily a bug in feature extraction.

Recommended “honest” evaluation options:

- **Speaker-based split**: train on OAF, test on YAF (or vice-versa).
- **Group split by speaker** (`GroupShuffleSplit` / `GroupKFold`) if you add speaker IDs per sample.
- **Phrase-based split** (if you can reliably extract phrase IDs) to prevent phrase leakage.

---

## Project structure

- `audio_analysis/`
	- `data_processing/feature_extraction.py` – mel-spectrogram feature extraction → pickle dataset
	- `train.py` – training script (stratified 70/30 random split) → `best_model.h5`
	- `inference.py` – preprocessing + prediction + Grad-CAM utilities
	- `evaluate_model.ipynb` – evaluation report + confusion matrix + Grad-CAM visualization
	- `inference_demo.ipynb` – interactive demo: prediction + TTS + Grad-CAM
- `data/` – TESS-style folders (e.g. `OAF_happy/`, `YAF_angry/`, …)
- `text_analysis/` – Whisper transcript extraction + TF-IDF/Naive Bayes notebook (separate track)

---

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure your data is present under `data/` (TESS folder layout).

---

## Audio feature extraction

This generates the feature pickle used by training/evaluation/inference.

```bash
python audio_analysis/data_processing/feature_extraction.py
```

Output:

- `audio_analysis/data_processing/tess_features.pkl`

Contents include:

- `features`: shape `(N, 128, 128, 1)`
- `labels`: integer labels
- `label_encoder`: sklearn `LabelEncoder`
- `emotion_list`: list of class names

---

## Train the CNN (audio)

```bash
python audio_analysis/train.py
```

Output:

- `audio_analysis/best_model.h5`

Notes:

- Training currently uses a **stratified random 70/30** split.
- This split is useful for sanity checks but may **overestimate real-world performance** on TESS.

---

## Evaluate the trained model

Open and run:

- `audio_analysis/evaluate_model.ipynb`

It reports:

- test accuracy / loss
- `classification_report`
- confusion matrix heatmap
- per-emotion accuracy bar plot
- Grad-CAM visualization for sample classes

---

## Inference demo (prediction + TTS + Grad-CAM)

Open and run:

- `audio_analysis/inference_demo.ipynb`

It will:

- load the trained model + label encoder
- pick an audio file (or you can set one)
- predict emotion + show confidence distribution
- generate in-memory TTS audio and play it in the notebook
- compute Grad-CAM and overlay it on the mel-spectrogram

---

## Text analysis track (why audio is needed)

The `text_analysis/` folder contains an experiment to see if **transcripts alone can capture emotion**:

- `extract_transcript.py` – uses OpenAI Whisper to transcribe TESS audio files
- `csv_generator.py` – generates `tess_metadata.csv` with Path, Speaker, Emotion, and Transcript columns
- `text_emotion_analysis.ipynb` – trains a Naive Bayes classifier on TF-IDF vectors of transcripts

### Key finding from text analysis

**Transcripts alone do NOT carry emotional information reliably.**

- The Naive Bayes text classifier achieves much lower accuracy than the audio CNN.
- This is because the TESS dataset uses the same **neutral utterance text** across all emotions—only the speaker's **delivery and prosody** (pitch, tone, speed, stress) differ.
- Whisper transcription discards all prosodic features, leaving only words that are identical across all emotions.
- Therefore, **audio-based features (mel-spectrograms) are essential** to capture emotion.

This validates our approach: **speech emotion ≠ word choice; it's about how you say it.**
