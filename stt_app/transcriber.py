
"""Local Faster-Whisper transcription engine."""

import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from stt_app.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Represent the result of speech transcription."""

    text: str
    language: str
    language_probability: float
    duration: float


class SpeechTranscriber:
    """Perform local speech recognition using Faster-Whisper."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the local speech recognition model."""

        self.settings = settings

        logger.info(
            "Loading Whisper model: %s | device: %s | compute: %s",
            settings.model_size,
            settings.device,
            settings.compute_type,
        )
        try:
            self.model = WhisperModel(
                model_size_or_path=settings.model_size,
                device=settings.device,
                compute_type=settings.compute_type,
            )

            logger.info("Whisper model loaded successfully.")

        except Exception as exc:
            logger.exception("Original Whisper initialization error")

            raise RuntimeError(
                f"Whisper initialization failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Path,
    ) -> TranscriptionResult:
        """
        Convert an audio recording into English text.

        Args:
            audio_path: Path to the recorded WAV file.

        Returns:
            Structured transcription result.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If the transcription is empty.
            RuntimeError: If transcription fails.
        """

        audio_path = Path(audio_path)

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file does not exist: {audio_path}"
            )

        try:
            logger.info("Starting transcription: %s", audio_path)

            segments, info = self.model.transcribe(
                str(audio_path),
                language="en",
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            # Faster-Whisper generates segments lazily.
            segment_texts = [
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ]

            transcript = " ".join(segment_texts).strip()

            if not transcript:
                raise ValueError(
                    "No recognizable English speech detected."
                )

            result = TranscriptionResult(
                text=transcript,
                language=info.language,
                language_probability=float(
                    info.language_probability
                ),
                duration=float(info.duration),
            )

            logger.info(
                "Transcription completed. Duration: %.2f seconds",
                result.duration,
            )

            return result

        except ValueError:
            raise

        except Exception as exc:
            logger.exception("Speech transcription failed.")

            raise RuntimeError(
                f"Unable to transcribe audio: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
