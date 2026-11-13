import sys

import ctranslate2
from faster_whisper import WhisperModel

print("Python:", sys.executable)
print("CTranslate2:", ctranslate2.__version__)

print("\nInitializing Whisper model...")

model = WhisperModel(
    "small.en",
    device="cpu",
    compute_type="int8",
)

print("\nSUCCESS: Whisper model initialized!")