import os
import json
import uuid
import hashlib
import datetime as dt
import math
import subprocess
import tempfile

import numpy as np
import librosa
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_V2_PATH = os.path.join(BASE_DIR, "models", "audio_deepfake_v2.keras")
MODEL_V4_PATH = os.path.join(BASE_DIR, "models", "audio_deepfake_v4.keras")
MODEL_V5_PATH = os.path.join(BASE_DIR, "models", "audio_deepfake_raw_v5.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
AUDIT_DIR = os.path.join(BASE_DIR, "audit_logs")
AUDIT_FILE = os.path.join(AUDIT_DIR, "security_events.jsonl")
SETTINGS_FILE = os.path.join(AUDIT_DIR, "security_settings.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
HOP_SECONDS = 1
N_MELS = 128
MAX_TIME_STEPS = 65
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "webm"}

DEFAULT_SETTINGS = {
    "medium_threshold": 40.0,
    "high_threshold": 75.0,
    "mfa_on_high": True,
    "callback_on_high": True,
    "protect_sensitive_transactions": True,
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

print("Loading V2 model...")
model_v2 = tf.keras.models.load_model(MODEL_V2_PATH)
print("Loading V4 model...")
model_v4 = tf.keras.models.load_model(MODEL_V4_PATH)
model_v5 = tf.keras.models.load_model(MODEL_V5_PATH) if os.path.exists(MODEL_V5_PATH) else None
print("Raw V5 loaded:", model_v5 is not None)
print("Primary V2 + V4 models loaded")


def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        settings.update(saved)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return settings


def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ffmpeg_convert_to_wav(source_path):
    """Convert browser WebM/Opus and other formats to mono 16-kHz WAV when ffmpeg is available."""
    ext = os.path.splitext(source_path)[1].lower()
    if ext == ".wav":
        return source_path, False
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return source_path, False

    converted = source_path + ".converted.wav"
    cmd = [ffmpeg, "-y", "-i", source_path, "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", converted]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if completed.returncode != 0 or not os.path.exists(converted):
        raise RuntimeError("Audio conversion failed. Please use WAV, FLAC, MP3, OGG, M4A or WebM with ffmpeg support.")
    return converted, True


def load_audio(audio_path):
    converted_path, converted = ffmpeg_convert_to_wav(audio_path)
    try:
        audio, _ = librosa.load(converted_path, sr=SAMPLE_RATE, mono=True)
    finally:
        if converted:
            try:
                os.remove(converted_path)
            except OSError:
                pass
    if audio is None or len(audio) == 0:
        raise ValueError("No usable audio was found in the recording.")
    return np.nan_to_num(audio).astype(np.float32)



def validate_speech_presence(audio):
    """Reject silence/empty captures before scoring or creating an audit event.

    This is a quality gate, not a deepfake classifier. It intentionally makes no
    claim about whether the detected sound is genuine speech.
    """
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.75:
        raise ValueError("The recording is too short for voice analysis. Capture at least 0.75 seconds of speech.")

    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    max_rms = float(np.max(rms)) if len(rms) else 0.0
    if peak < 0.003 or max_rms < 0.003:
        raise ValueError("No audible speech was detected. No risk score or security event was created.")

    # Frames need enough absolute energy to avoid treating digital silence and
    # very low-level microphone noise as a voice sample.
    active_frames = int(np.sum(rms >= max(0.006, max_rms * 0.12)))
    active_seconds = active_frames * 256 / SAMPLE_RATE
    if active_seconds < 0.45:
        raise ValueError("Not enough audible speech was detected. No risk score or security event was created.")
    return {"duration": round(duration, 2), "active_speech_seconds": round(active_seconds, 2)}


def audio_to_features(audio):
    target_length = SAMPLE_RATE * WINDOW_SECONDS
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    audio = np.clip(audio, -1.0, 1.0)
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < MAX_TIME_STEPS:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_TIME_STEPS - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_TIME_STEPS]
    return mel_db[np.newaxis, ..., np.newaxis].astype(np.float32)



def raw_v5_spoof_probability(segment):
    if model_v5 is None:
        return None
    features = audio_to_features(segment)
    real_probability = float(model_v5.predict(features, verbose=0)[0][0])
    return 1.0 - real_probability

def cosine_similarity(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def acoustic_signature(audio):
    """Compact, non-biometric acoustic signature for optional cross-session consistency."""
    if len(audio) < SAMPLE_RATE:
        audio = np.pad(audio, (0, SAMPLE_RATE - len(audio)))
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    vec = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1)])
    vec = np.nan_to_num(vec).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def prosody_analysis(audio):
    duration = len(audio) / SAMPLE_RATE
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=256)[0]
    try:
        f0 = librosa.yin(audio, fmin=70, fmax=350, sr=SAMPLE_RATE, frame_length=2048, hop_length=256)
        voiced = f0[np.isfinite(f0) & (f0 > 0)]
    except Exception:
        voiced = np.array([])

    energy_db = librosa.amplitude_to_db(np.maximum(rms, 1e-7), ref=np.max)
    silence_ratio = float(np.mean(energy_db < -35)) if len(energy_db) else 0.0
    pitch_variation = float(np.std(voiced) / max(np.mean(voiced), 1e-6)) if len(voiced) else 0.0
    energy_variation = float(np.std(rms) / max(np.mean(rms), 1e-6)) if len(rms) else 0.0
    speaking_activity = 1.0 - silence_ratio

    # These are interpretable acoustic indicators, not a standalone deepfake detector.
    stability = 100.0 * np.clip(
        0.45 * min(pitch_variation / 0.35, 1.0) +
        0.35 * min(energy_variation / 1.2, 1.0) +
        0.20 * speaking_activity,
        0, 1
    )

    return {
        "pitch_variation": round(pitch_variation, 3),
        "energy_variation": round(energy_variation, 3),
        "pause_ratio": round(silence_ratio * 100, 1),
        "speaking_activity": round(speaking_activity * 100, 1),
        "prosody_activity_score": round(stability, 1),
        "duration": round(duration, 2),
    }


def split_segments(audio):
    total_samples = len(audio)
    window_samples = SAMPLE_RATE * WINDOW_SECONDS
    hop_samples = SAMPLE_RATE * HOP_SECONDS
    if total_samples <= window_samples:
        return [(0, audio)]
    segments = []
    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment = audio[start:end]
        if len(segment) >= SAMPLE_RATE:
            segments.append((start, segment))
        start += hop_samples
    return segments or [(0, audio)]


def merge_timeline(timeline):
    """Merge overlapping hop windows into continuous, non-overlapping display regions."""
    if not timeline:
        return []
    boundaries = sorted(set([0.0] + [x["start"] for x in timeline] + [x["end"] for x in timeline]))
    regions = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        active = [x for x in timeline if x["start"] < b and x["end"] > a]
        if not active:
            continue
        spoof = float(np.mean([x["smoothed_spoof_probability"] for x in active]))
        label = "SPOOF" if spoof >= 50.0 else "REAL"
        if regions and regions[-1]["label"] == label and abs(regions[-1]["end"] - a) < 1e-6:
            regions[-1]["end"] = round(b, 2)
        else:
            regions.append({"start": round(a, 2), "end": round(b, 2), "label": label})
    return regions


def analyze_audio(audio, reference_audio=None, context=None):
    segments = split_segments(audio)
    raw_scores = []
    v5_scores = []
    timeline = []
    for start_sample, segment in segments:
        features = audio_to_features(segment)
        v2_spoof = 1.0 - float(model_v2.predict(features, verbose=0)[0][0])
        v4_spoof = 1.0 - float(model_v4.predict(features, verbose=0)[0][0])
        spoof_probability = 0.50 * v2_spoof + 0.50 * v4_spoof
        v5_spoof = raw_v5_spoof_probability(segment)
        raw_scores.append(spoof_probability)
        if v5_spoof is not None: v5_scores.append(v5_spoof)
        st = start_sample / SAMPLE_RATE; en = min(st + WINDOW_SECONDS, len(audio) / SAMPLE_RATE)
        timeline.append({"start":round(st,2),"end":round(en,2),"label":"SPOOF" if spoof_probability>=0.50 else "REAL","spoof_probability":round(spoof_probability*100,2),"v2_spoof_probability":round(v2_spoof*100,2),"v4_spoof_probability":round(v4_spoof*100,2),"v5_secondary_spoof_probability":round(v5_spoof*100,2) if v5_spoof is not None else None})
    raw=np.asarray(raw_scores,dtype=np.float32)
    smoothed=np.array([np.mean(raw[max(0,i-1):min(len(raw),i+2)]) for i in range(len(raw))])
    for item,score in zip(timeline,smoothed): item["smoothed_spoof_probability"]=round(float(score*100),2); item["label"]="SPOOF" if score>=0.50 else "REAL"
    overall=float(np.mean(raw)); mx=float(np.max(raw)); model_risk=(0.65*overall+0.35*mx)*100
    v5_mean=float(np.mean(v5_scores)) if v5_scores else None
    disagreement=bool(v5_mean is not None and model_risk<40 and v5_mean>=0.60)
    context=context or {}; settings=load_settings(); adj=0.0; reasons=[]
    if context.get("unknown_caller"): adj+=8; reasons.append("unknown caller")
    if context.get("sensitive_transaction") and settings["protect_sensitive_transactions"]: adj+=12; reasons.append("sensitive transaction")
    if context.get("first_contact"): adj+=5; reasons.append("first contact")
    if context.get("previous_risk")=="HIGH": adj+=10; reasons.append("previous high-risk event")
    risk=min(100.0,model_risk+adj)
    if risk>=float(settings["high_threshold"]):
        level="HIGH"
        safeguards=[]
        if settings["callback_on_high"]: safeguards.append("an independent callback")
        if settings["mfa_on_high"]: safeguards.append("MFA")
        verification=" and ".join(safeguards) if safeguards else "manual identity verification"
        rec=f"Do not authorize sensitive requests. Require {verification} before proceeding."
        action="BLOCK / VERIFY"
    elif risk>=float(settings["medium_threshold"]): level="MEDIUM"; rec="Pause sensitive actions and perform additional identity verification."; action="STEP-UP VERIFICATION"
    else: level="LOW"; rec="No strong synthetic characteristics detected by the primary V2+V4 detector. Continue normal security verification."; action="ALLOW WITH NORMAL CHECKS"
    if disagreement: rec="Primary V2+V4 is relatively low-risk, but Raw V5 reports strong spoof evidence. Perform independent callback and/or MFA before sensitive actions."; action="MODEL DISAGREEMENT / VERIFY"
    ref_score=None
    if reference_audio is not None: ref_score=round(max(0,min(1,(cosine_similarity(acoustic_signature(audio),acoustic_signature(reference_audio))+1)/2))*100,1)
    return {"duration":round(len(audio)/SAMPLE_RATE,2),"risk_score":round(risk,2),"model_risk_score":round(model_risk,2),"risk_level":level,"overall_spoof_probability":round(overall*100,2),"max_spoof_probability":round(mx*100,2),"v2_spoof_probability":round(np.mean([x["v2_spoof_probability"] for x in timeline]),2),"v4_spoof_probability":round(np.mean([x["v4_spoof_probability"] for x in timeline]),2),"v5_secondary_spoof_probability":round(v5_mean*100,2) if v5_mean is not None else None,"v5_disagreement":disagreement,"primary_detector":"V2 + V4 (50/50)","timeline":merge_timeline(timeline),"segments":timeline,"prosody":prosody_analysis(audio),"speaker_consistency_score":ref_score,"speaker_consistency_available":ref_score is not None,"context_adjustment":round(adj,2),"context_reasons":reasons,"recommendation":rec,"recommended_action":action,"response_requirements":{"mfa":bool(settings["mfa_on_high"] and level == "HIGH"),"callback":bool(settings["callback_on_high"] and level == "HIGH"),"sensitive_protection":bool(settings["protect_sensitive_transactions"]),"configured":True},"thresholds":{"medium":float(settings["medium_threshold"]),"high":float(settings["high_threshold"])} ,"analysis_note":"Primary risk uses the V2+V4 ensemble. Raw V5 is a secondary disagreement signal and does not directly raise the primary risk score."}


def read_events(limit=100):
    events = []
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return list(reversed(events[-limit:]))


def create_audit_event(filename, result, source_type, context):
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    previous_hash = "GENESIS"
    events = read_events(limit=1)
    if events:
        previous_hash = events[0].get("event_hash", "GENESIS")

    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "source_type": source_type,
        "filename": filename,
        "duration": result["duration"],
        "risk_score": result["risk_score"],
        "model_risk_score": result["model_risk_score"],
        "risk_level": result["risk_level"],
        "spoof_probability": result["overall_spoof_probability"],
        "context_adjustment": result["context_adjustment"],
        "previous_hash": previous_hash,
        "raw_audio_retained": False,
        "language_profile": context.get("language_profile", "Language-agnostic"),
        "contact_context": context.get("contact_context", ""),
        "transaction_context": context.get("transaction_context", ""),
    }
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    event["event_hash"] = hashlib.sha256(payload).hexdigest()
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def verify_audit_chain():
    events = []
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return {"valid": True, "checked": 0, "message": "No audit events yet."}

    # Legacy events created by the earlier app are reported but not retroactively rewritten.
    prev = "GENESIS"
    checked = 0
    for event in events:
        if "previous_hash" not in event:
            prev = event.get("event_hash", prev)
            continue
        if event.get("previous_hash") != prev:
            return {"valid": False, "checked": checked, "message": "Audit chain mismatch detected."}
        copy = dict(event)
        stored = copy.pop("event_hash", None)
        payload = json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != stored:
            return {"valid": False, "checked": checked, "message": "Audit event hash mismatch detected."}
        prev = stored
        checked += 1
    return {"valid": True, "checked": checked, "message": "Chained audit events verified."}


def parse_context(form):
    return {
        "language_profile": form.get("language_profile", "Language-agnostic"),
        "contact_context": form.get("contact_context", ""),
        "transaction_context": form.get("transaction_context", ""),
        "unknown_caller": form.get("unknown_caller") == "true",
        "sensitive_transaction": form.get("sensitive_transaction") == "true",
        "first_contact": form.get("first_contact") == "true",
        "previous_risk": form.get("previous_risk", "LOW"),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "online", "models": {"v2": True, "v4": True, "v5": model_v5 is not None}})


@app.route("/predict", methods=["POST"])
def predict():
    audio_file = request.files.get("audio")
    reference_file = request.files.get("reference_audio")
    source_type = request.form.get("source_type", "recorded")
    if not audio_file or not audio_file.filename:
        return jsonify({"success": False, "error": "No audio file supplied."}), 400
    if not allowed_file(audio_file.filename):
        return jsonify({"success": False, "error": "Unsupported audio format."}), 400

    saved_path = None
    ref_path = None
    try:
        safe_name = secure_filename(audio_file.filename) or "voice_input.wav"
        saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe_name}")
        audio_file.save(saved_path)
        audio = load_audio(saved_path)
        validate_speech_presence(audio)

        reference_audio = None
        if reference_file and reference_file.filename and allowed_file(reference_file.filename):
            safe_ref = secure_filename(reference_file.filename) or "reference.wav"
            ref_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{safe_ref}")
            reference_file.save(ref_path)
            reference_audio = load_audio(ref_path)

        context = parse_context(request.form)
        result = analyze_audio(audio, reference_audio, context)
        event = create_audit_event(safe_name, result, source_type, context)
        return jsonify({
            "success": True,
            "audit_event_id": event["event_id"],
            "audit_hash": event["event_hash"],
            "privacy": {"raw_audio_retained": False, "raw_audio_deleted_after_analysis": True},
            "result": result,
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"success": False, "error": str(exc)}), 500
    finally:
        for path in (saved_path, ref_path):
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.route("/api/events")
def api_events():
    return jsonify({"success": True, "events": read_events()})


@app.route("/api/audit/verify")
def api_audit_verify():
    return jsonify(verify_audit_chain())


@app.route("/api/privacy")
def api_privacy():
    events = read_events()
    verification = verify_audit_chain()
    return jsonify({
        "success": True,
        "raw_audio_retained": False,
        "reference_audio_retained": False,
        "audit_event_count": len(events),
        "audit_chain_valid": verification["valid"],
        "checked_events": verification["checked"],
        "last_event_at": events[0].get("timestamp") if events else None,
    })


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "GET":
        return jsonify({"success": True, "settings": load_settings()})
    incoming = request.get_json(silent=True) or {}
    settings = load_settings()
    for key in DEFAULT_SETTINGS:
        if key in incoming:
            settings[key] = incoming[key]
    try:
        settings["medium_threshold"] = float(max(0, min(99, float(settings["medium_threshold"]))))
        settings["high_threshold"] = float(max(settings["medium_threshold"] + 1, min(100, float(settings["high_threshold"]))))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Thresholds must be numeric."}), 400
    for key in ("mfa_on_high", "callback_on_high", "protect_sensitive_transactions"):
        settings[key] = bool(settings[key])
    save_settings(settings)
    return jsonify({"success": True, "settings": settings})


@app.route("/api/security-action", methods=["POST"])
def security_action():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "VERIFY")
    event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event_type": "SECURITY_WORKFLOW",
        "action": action,
        "risk_level": data.get("risk_level", "UNKNOWN"),
        "source_event_id": data.get("source_event_id", ""),
        "previous_hash": "GENESIS",
        "raw_audio_retained": False,
    }
    events = read_events(limit=1)
    if events:
        event["previous_hash"] = events[0].get("event_hash", "GENESIS")
    payload = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    event["event_hash"] = hashlib.sha256(payload).hexdigest()
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")
    return jsonify({"success": True, "event": event})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
