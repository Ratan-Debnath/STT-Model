
from faster_whisper import WhisperModel


def main() -> None:
    audio_path = (
        r"C:\Users\Think Pad\PycharmProjects"
        r"\STT_model_IELTS_v1\stt_app\recordings"
        r"\recording_20261109_070517.wav"
    )

    print("Initializing Faster-Whisper...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    print("Model initialized successfully.")
    print("Starting transcription with VAD...")

    segments, info = model.transcribe(
        audio_path,
        vad_filter=True,
    )

    print("Language:", info.language)

    for segment in segments:
        print(
            f"[{segment.start:.2f}s - "
            f"{segment.end:.2f}s] {segment.text}"
        )

    print("TRANSCRIPTION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
