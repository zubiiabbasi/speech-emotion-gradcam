"""
Train the TESS mel-spectrogram CNN with strict splits and on-the-fly augmentation.

Pipeline
    Load ``tess_features.pkl`` → split (default: **train = OAF only**, **val =
    YAF only** — held-out speaker for both validation metrics and notebook test)
    → save ``tess_eval_split.npz`` → build model → ``tf.data`` train set
    (shuffle + ``augment_mel_spec`` each epoch) / val set (YAF, no aug) →
    ``model.fit`` with checkpoint, early stopping, and LR reduction.

Environment
    ``TESS_SPLIT_MODE``: ``speaker`` (default) | ``sentence_group`` | ``random``.

Outputs
    ``best_model.h5`` (project cwd), ``tess_eval_split.npz`` beside the features pickle.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models import build_emotion_cnn, compile_model

# "speaker" | "sentence_group" | "random"
SPLIT_MODE = os.environ.get("TESS_SPLIT_MODE", "speaker").strip().lower()


def _resolve_features_path(features_file="data_processing/tess_features.pkl"):
    possible_paths = [
        Path(features_file),
        Path("audio_analysis") / features_file,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def load_features(features_file="data_processing/tess_features.pkl"):
    """Load pickle; returns (features, labels, label_encoder, emotion_list, speakers, sentence_groups, path) or Nones."""
    features_path = _resolve_features_path(features_file)

    if features_path is None:
        print("ERROR: Could not find features pickle file!")
        print("Tried:")
        for path in [Path(features_file), Path("audio_analysis") / features_file]:
            print(f"  - {path}")
        print("\nFirst run: python audio_analysis/data_processing/feature_extraction.py")
        return (None,) * 7

    try:
        with open(features_path, "rb") as f:
            data = pickle.load(f)

        features = data["features"]
        labels = data["labels"]
        label_encoder = data["label_encoder"]
        emotion_list = data["emotion_list"]
        speakers = data.get("speakers")
        sentence_groups = data.get("sentence_groups")

        print("Features loaded successfully!")
        print(f"  Path: {features_path}")
        print(f"  Shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Emotions: {emotion_list}\n")

        if speakers is None:
            print(
                "ERROR: Pickle has no 'speakers' array. Re-run:\n"
                "  python audio_analysis/data_processing/feature_extraction.py\n"
            )
            return (None,) * 7

        return features, labels, label_encoder, emotion_list, speakers, sentence_groups, features_path

    except Exception as e:
        print(f"Error loading features: {e}")
        return (None,) * 7


def prepare_data(
    features,
    labels,
    speakers,
    sentence_groups,
    split_mode=SPLIT_MODE,
    test_size=0.3,
    random_state=42,
):
    """
    Build train/test tensors. Default: speaker-independent (OAF train, YAF test).
    """
    speakers = np.asarray(speakers)
    labels = np.asarray(labels)
    n = len(labels)

    if split_mode == "speaker":
        oaf_mask = speakers == "OAF"
        yaf_mask = speakers == "YAF"
        if not (oaf_mask.any() and yaf_mask.any()):
            raise ValueError("Speaker split needs both OAF and YAF in the dataset.")
        train_idx = np.flatnonzero(oaf_mask)
        test_idx = np.flatnonzero(yaf_mask)
        print("Data split: speaker-independent (TESS)")
        print("  Train: OAF only")
        print("  Test:  YAF only\n")

    elif split_mode == "sentence_group":
        if sentence_groups is None:
            raise ValueError("sentence_group split needs 'sentence_groups' in the pickle.")
        groups = np.asarray(sentence_groups)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(np.zeros((n, 1)), labels, groups=groups))
        train_idx = train_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)
        print("Data split: sentence_group (GroupShuffleSplit, no shared sentence across train/test)")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Test samples:  {len(test_idx)}\n")

    elif split_mode == "random":
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        train_idx = train_idx.astype(np.int64)
        test_idx = test_idx.astype(np.int64)
        print("Data split: random stratified (leaky on TESS — for comparison only)")
        print(f"  Train samples: {len(train_idx)} ({100 * (1 - test_size):.0f}%)")
        print(f"  Test samples:  {len(test_idx)} ({100 * test_size:.0f}%)\n")

    else:
        raise ValueError(f"Unknown TESS_SPLIT_MODE: {split_mode!r}")

    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}\n")

    return X_train, X_test, y_train, y_test, train_idx, test_idx


def save_eval_split(features_path: Path, train_idx, test_idx, split_mode: str):
    out = features_path.with_name("tess_eval_split.npz")
    np.savez(out, train_idx=train_idx, test_idx=test_idx, split_mode=np.array(split_mode))
    print(f"Saved eval split indices to: {out}\n")


def load_eval_split(features_path: Path):
    """Load train/test index arrays written by training."""
    out = features_path.with_name("tess_eval_split.npz")
    if not out.exists():
        raise FileNotFoundError(
            f"Missing {out}. Run train.py once so it saves the same split used in training."
        )
    z = np.load(out, allow_pickle=True)
    return z["train_idx"].astype(np.int64), z["test_idx"].astype(np.int64), str(z["split_mode"].item())


def load_train_test_for_eval(features_file="data_processing/tess_features.pkl"):
    """
    For notebooks: load features pickle + tess_eval_split.npz (same rows as training).
    Returns X_train, X_test, y_train, y_test, label_encoder, emotion_list.
    """
    path = _resolve_features_path(features_file)
    if path is None:
        raise FileNotFoundError("Could not find tess_features.pkl")

    with open(path, "rb") as f:
        data = pickle.load(f)

    features = data["features"]
    labels = data["labels"]
    label_encoder = data["label_encoder"]
    emotion_list = data["emotion_list"]

    train_idx, test_idx, _ = load_eval_split(path)
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    return X_train, X_test, y_train, y_test, label_encoder, emotion_list


def augment_mel_spec(mel_spec):
    """Gentler SpecAugment-style masks + noise (mel layout: freq x time x ch)."""
    aug = np.array(mel_spec, copy=True, dtype=np.float32)

    t_mask = np.random.randint(2, 12)
    t_start = np.random.randint(0, 128 - t_mask)
    aug[:, t_start : t_start + t_mask, :] = 0.0

    f_mask = np.random.randint(2, 10)
    f_start = np.random.randint(0, 128 - f_mask)
    aug[f_start : f_start + f_mask, :, :] = 0.0

    aug += np.random.normal(0.0, 0.01, aug.shape).astype(np.float32)
    return aug


def make_dataset(X, y, batch_size, augment=False, shuffle=False):
    """tf.data pipeline; augment=True applies a fresh random mask each epoch (reshuffle_each_iteration)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    n = len(X)
    h, w, c = X.shape[1], X.shape[2], X.shape[3]

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(n, 1), reshuffle_each_iteration=True)

    if augment:

        def aug_map(x, lbl):
            x_aug = tf.numpy_function(augment_mel_spec, [x], tf.float32)
            x_aug.set_shape([h, w, c])
            return x_aug, lbl

        ds = ds.map(aug_map, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def create_callbacks(model_name="best_model.h5"):
    """Checkpoint on val_accuracy; early stop patience 25; ReduceLROnPlateau patience 8, min_lr 1e-6."""
    return [
        ModelCheckpoint(
            filepath=model_name,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def train_model(model, train_ds, val_ds, epochs=100):
    """Fit ``model`` on ``train_ds`` / ``val_ds`` (tf.data); uses ``create_callbacks``."""
    print(f"\n{'=' * 70}")
    print("TRAINING EMOTION RECOGNITION CNN")
    print(f"{'=' * 70}")
    print(f"Epochs: {epochs}")
    print("Train: tf.data with on-the-fly augmentation each epoch")
    print("Val:   tf.data, no augmentation")
    print(f"{'=' * 70}\n")

    callbacks = create_callbacks(model_name="best_model.h5")

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )


def main():
    print(f"\n{'=' * 70}")
    print("EMOTION RECOGNITION CNN - TRAINING")
    print(f"{'=' * 70}\n")

    features, labels, label_encoder, emotion_list, speakers, sentence_groups, features_path = load_features()
    if features is None:
        return

    X_train, X_test, y_train, y_test, train_idx, test_idx = prepare_data(
        features, labels, speakers, sentence_groups, split_mode=SPLIT_MODE
    )
    save_eval_split(features_path, train_idx, test_idx, SPLIT_MODE)

    print("Building CNN model...")
    model = build_emotion_cnn(input_shape=(128, 128, 1), num_classes=len(emotion_list))
    model = compile_model(model, learning_rate=0.0003)
    print(f"Model built with {model.count_params():,} parameters\n")

    batch_size = 32
    train_ds = make_dataset(
        X_train, y_train, batch_size=batch_size, augment=True, shuffle=True
    )
    val_ds = make_dataset(
        X_test, y_test, batch_size=batch_size, augment=False, shuffle=False
    )
    print(f"Train samples per epoch: {len(X_train)} (fresh augmentation each pass)\n")

    train_model(model, train_ds, val_ds, epochs=100)


if __name__ == "__main__":
    main()
