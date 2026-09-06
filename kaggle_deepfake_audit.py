"""Kaggle notebook-ready audit/evaluation harness for VoiceShield.

Run this file with `%run /kaggle/working/kaggle_deepfake_audit.py` after placing
the V2/V4/V5 checkpoints in /kaggle/working.  It does not train another raw
model by default: Raw V5 must first pass the promotion gate on ASVspoof dev.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

SEED = 42
SR, SECONDS, N_MELS, FRAMES = 16000, 3, 128, 65
ASV_ROOT = Path("/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA")
LIBRI_ROOT = Path("/kaggle/input/datasets/victorling/librispeech-clean/LibriSpeech")
IIIT_ROOT = Path("/kaggle/input/datasets/sizlingdhairya1/iiit-spoken-language-datasets/IIIT Spoken Language Datasets")
EXTERNAL_FAKE = Path("/kaggle/input/datasets/varunadhithya/fake-voice/DF_E_2000137.wav")
WORK = Path("/kaggle/working")

random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)


def read_protocol(protocol: Path, audio_dir: Path, split: str) -> pd.DataFrame:
    """Read ASVspoof labels strictly from the protocol; never from filenames."""
    rows = []
    for line in protocol.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        utt, label = fields[1], fields[-1].lower()
        if label not in {"bonafide", "spoof"}:
            raise ValueError(f"Unexpected protocol label {label!r} in {protocol}")
        rows.append({"path": str(audio_dir / f"{utt}.flac"), "label": label,
                     "y": int(label == "spoof"), "split": split,
                     "speaker": fields[0], "attack": fields[-2] if len(fields) > 4 else "unknown"})
    frame = pd.DataFrame(rows)
    missing = frame.loc[~frame.path.map(lambda x: Path(x).exists())]
    if not missing.empty:
        raise FileNotFoundError(f"{len(missing)} protocol files are missing, e.g. {missing.iloc[0].path}")
    return frame


def asvspoof_manifests():
    train = read_protocol(ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
                          ASV_ROOT / "ASVspoof2019_LA_train/flac", "train")
    dev = read_protocol(ASV_ROOT / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
                        ASV_ROOT / "ASVspoof2019_LA_dev/flac", "dev")
    assert len(train) == 25380, f"Expected 25,380 train records, found {len(train)}"
    assert train.y.sum() == 22800, "ASVspoof train class counts do not match the protocol"
    return train, dev


def genuine_manifests(max_libri=20000, per_iiit_language=1000):
    """Build genuine-only manifests; these samples are never labelled spoof."""
    libri = sorted(LIBRI_ROOT.glob("train-clean-*/*/*/*.flac"))
    rng = random.Random(SEED); rng.shuffle(libri)
    libri_rows = [{"path": str(p), "label": "bonafide", "y": 0, "split": "aux_train",
                   "speaker": p.parts[-3], "attack": "genuine_librispeech"} for p in libri[:max_libri]]
    rows = []
    for language in ("Tamil", "Hindi", "Bengali", "Kannada", "Telugu", "Marathi", "Malayalam"):
        files = [p for p in IIIT_ROOT.joinpath(language).rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"}]
        rng.shuffle(files)
        rows.extend({"path": str(p), "label": "bonafide", "y": 0, "split": "aux_train",
                     "speaker": language, "attack": f"genuine_iiit_{language.lower()}"}
                    for p in files[:per_iiit_language])
    manifest = pd.DataFrame(libri_rows + rows)
    language_counts = manifest.loc[manifest.attack.str.startswith("genuine_iiit_"), "attack"].value_counts().to_dict()
    print("IIIT genuine samples discovered:", language_counts)
    missing = [lang for lang in ("tamil", "hindi", "bengali", "kannada", "telugu", "marathi", "malayalam")
               if language_counts.get(f"genuine_iiit_{lang}", 0) < per_iiit_language]
    if missing:
        print("WARNING: fewer than the requested IIIT samples were found for:", ", ".join(missing))
    return manifest


def mel_windows(path: str):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    window, hop = SR * SECONDS, SR
    if len(audio) <= window:
        starts = [0]
    else:
        starts = list(range(0, len(audio) - window + 1, hop))
        if starts[-1] != len(audio) - window: starts.append(len(audio) - window)
    features = []
    for start in starts:
        clip = audio[start:start + window]
        clip = np.pad(clip, (0, max(0, window - len(clip))))[:window]
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=clip, sr=SR, n_mels=N_MELS), ref=np.max)
        mel = np.pad(mel[:, :FRAMES], ((0, 0), (0, max(0, FRAMES - mel.shape[1]))))
        features.append(mel[..., None].astype("float32"))
    return np.stack(features)


def spoof_scores(model, paths, batch_windows=256):
    """Mean all overlapping 3-second windows using GPU-sized prediction batches.

    Scores are still aggregated per file; batching changes throughput, not the
    metric or decision rule.  It avoids one Keras predict call per dev file.
    """
    paths = list(paths)
    per_file = [[] for _ in paths]
    pending, owners = [], []

    def flush():
        if not pending:
            return
        batch = np.concatenate(pending, axis=0)
        real = model.predict(batch, batch_size=batch_windows, verbose=0).reshape(-1)
        for owner, score in zip(owners, real):
            per_file[owner].append(1.0 - float(score))
        pending.clear(); owners.clear()

    window_count = 0
    for index, path in enumerate(paths):
        windows = mel_windows(path)
        pending.append(windows)
        owners.extend([index] * len(windows))
        window_count += len(windows)
        if sum(len(item) for item in pending) >= batch_windows:
            flush()
        if (index + 1) % 250 == 0:
            print(f"Scored {index + 1:,}/{len(paths):,} files ({window_count:,} windows)")
    flush()
    if any(not scores for scores in per_file):
        raise RuntimeError("At least one evaluation file produced no feature windows.")
    return np.asarray([float(np.mean(scores)) for scores in per_file])


def threshold_at_fpr(y, scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y, scores)
    valid = np.flatnonzero(fpr <= target_fpr)
    if not len(valid): return float("inf")
    return float(thresholds[valid[np.argmax(tpr[valid])]])


def report(name, y, scores, threshold, groups=None):
    pred = scores >= threshold
    fpr = float(np.mean(pred[y == 0])) if np.any(y == 0) else float("nan")
    tpr = float(np.mean(pred[y == 1])) if np.any(y == 1) else float("nan")
    result = {"name": name, "n": int(len(y)), "roc_auc": float(roc_auc_score(y, scores)),
              "average_precision": float(average_precision_score(y, scores)), "threshold": float(threshold),
              "fpr": fpr, "tpr": tpr, "positive_rate": float(np.mean(pred))}
    if groups is not None:
        result["by_attack"] = {str(g): float(np.mean(pred[np.asarray(groups) == g])) for g in sorted(set(groups))}
    print(json.dumps(result, indent=2))
    return result


def promotion_gate(ensemble, raw_v5, tolerance=0.02):
    """Raw is eligible only when it matches primary discrimination and calibration behaviour.

    This deliberately rejects the known V5 'everything is spoof' failure mode.
    A pass is permission to run a separate candidate experiment, not permission
    to deploy the candidate as the primary detector.
    """
    rules = {
        "not_degenerate": raw_v5["positive_rate"] < 0.98 and raw_v5["positive_rate"] > 0.02,
        "auc_not_worse": raw_v5["roc_auc"] >= ensemble["roc_auc"] - tolerance,
        "recall_not_worse": raw_v5["tpr"] >= ensemble["tpr"] - tolerance,
        "fpr_not_worse": raw_v5["fpr"] <= ensemble["fpr"] + tolerance,
    }
    print("RAW MODEL PROMOTION GATE:", rules, "PASS" if all(rules.values()) else "FAIL")
    return all(rules.values())


def audit_existing_models():
    train, dev = asvspoof_manifests()
    # Auxiliary genuine data can be used only in training experiments, never to
    # relabel ASVspoof or manufacture synthetic multilingual examples.
    auxiliary = genuine_manifests()
    print({"asv_train": train.label.value_counts().to_dict(), "asv_dev": dev.label.value_counts().to_dict(),
           "auxiliary_genuine_only": len(auxiliary)})

    v2 = tf.keras.models.load_model(WORK / "audio_deepfake_v2.keras", compile=False)
    v4 = tf.keras.models.load_model(WORK / "audio_deepfake_v4.keras", compile=False)
    v5_path = WORK / "audio_deepfake_raw_v5.keras"
    s2, s4 = spoof_scores(v2, dev.path), spoof_scores(v4, dev.path)
    ensemble_scores = (s2 + s4) / 2
    # Threshold selection must happen on a calibration partition, not this final
    # dev report.  Until such a held-out calibration set is supplied, report a
    # transparent operating point rather than claiming final calibration.
    operating_threshold = 0.50
    ensemble = report("V2+V4 primary / ASVspoof dev", dev.y.to_numpy(), ensemble_scores,
                      operating_threshold, dev.attack.to_numpy())

    raw_ok = False
    if v5_path.exists():
        v5 = tf.keras.models.load_model(v5_path, compile=False)
        raw = report("Raw V5 secondary / ASVspoof dev", dev.y.to_numpy(), spoof_scores(v5, dev.path),
                     operating_threshold, dev.attack.to_numpy())
        raw_ok = promotion_gate(ensemble, raw)
    else:
        print("Raw V5 checkpoint absent: raw training and promotion are skipped.")

    if EXTERNAL_FAKE.exists():
        print("External fake challenge score (not used for fitting or threshold selection):",
              float(np.mean(spoof_scores(v2, [str(EXTERNAL_FAKE)]) + spoof_scores(v4, [str(EXTERNAL_FAKE)])) / 2))
    if not raw_ok:
        print("Decision: retain V2+V4 as primary and V5 as secondary only. Do not train/promote a final raw model.")
    return {"primary": ensemble, "raw_eligible_for_experiment": raw_ok}


if __name__ == "__main__":
    audit_existing_models()
