VOICESHIELD AI SECURITY
========================

1. Extract this ZIP anywhere.
2. Open the extracted deepfake_voice_security folder.
3. Double-click setup.bat ONCE.
4. Wait for installation to finish.
5. Double-click run.bat.
6. Open http://127.0.0.1:5000

IMPORTANT:
The Python environment is deliberately created in:
%LOCALAPPDATA%\VoiceShieldVenv

This avoids the Windows TensorFlow long-folder-path problem even if the
project itself is inside Downloads or another long folder path.

Do NOT create or use a venv inside this project folder.

FINAL MODEL LOGIC: V2+V4 is the primary detector (50/50). Raw V5 is a secondary disagreement warning. Place audio_deepfake_raw_v5.keras in models/ to enable it.
