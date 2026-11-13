
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel


def main() -> None:
    audio_path = Path(
        r"C:\Users\Think Pad\PycharmProjects"
        r"\STT_model_IELTS_v1\stt_app\recordings"
        r"\recording_20261109_070517.wav"
    )

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    audio, sample_rate = sf.read(audio_path)

    duration = len(audio) / sample_rate

    print("Audio diagnostics")
    print("-----------------")
    print("File size:", audio_path.stat().st_size, "bytes")
    print("Sample rate:", sample_rate)
    print("Duration:", round(duration, 2), "seconds")
    print("Audio shape:", audio.shape)

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("\nTranscription WITHOUT VAD:")

    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=False,
        beam_size=5,
    )

    results = list(segments)

    print("Detected language:", info.language)
    print("Number of segments:", len(results))

    for segment in results:
        print(
            f"[{segment.start:.2f}s - "
            f"{segment.end:.2f}s] {segment.text}"
        )

    if not results:
        print("WARNING: No speech segments detected.")


if __name__ == "__main__":
    main()
