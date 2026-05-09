"""Data processing: TESS mel extraction and normalization helpers."""
from .feature_extraction import extract_mel_spectrogram, normalize_mel_db, process_tess_dataset

__all__ = ['extract_mel_spectrogram', 'normalize_mel_db', 'process_tess_dataset']
