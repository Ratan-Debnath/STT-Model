"""Application entry point for local speech recognition."""

import logging

from stt_app.audio_recorder import MicrophoneRecorder
from stt_app.config import Settings
from stt_app.transcriber import SpeechTranscriber


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Execute the complete local STT pipeline."""
    configure_logging()

    logger = logging.getLogger(__name__)

    try:
        settings = Settings.from_environment()

        print("\n====================================")
        print(" IELTS LOCAL SPEECH-TO-TEXT SYSTEM")
        print("====================================")

        recorder = MicrophoneRecorder(settings)

        # Record speech from the microphone.
        audio_path = recorder.record()

        # Initialize the AI model after recording.
        print("\nLoading speech recognition model...")

        transcriber = SpeechTranscriber(settings)

        # Convert speech into text.
        result = transcriber.transcribe(audio_path)

        print("\n====================================")
        print(" TRANSCRIPTION RESULT")
        print("====================================")

        print(f"\nRecognized text:\n{result.text}")

        print(f"\nLanguage: {result.language}")
        print(
            "Language confidence: "
            f"{result.language_probability:.2%}"
        )
        print(f"Audio duration: {result.duration:.2f} seconds")

        print("\n====================================")
        print(" TRANSCRIPTION COMPLETED")
        print("====================================")

    except KeyboardInterrupt:
        logger.warning("Application interrupted by user.")

    except Exception:
        logger.exception("Application execution failed.")
        raise


if __name__ == "__main__":
    main()