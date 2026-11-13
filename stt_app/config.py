
"""Application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application configuration."""

    model_size: str = "small.en"
    device: str = "cpu"
    compute_type: str = "int8"
    sample_rate: int = 16000
    audio_directory: Path = Path("recordings")

    # None means automatically discover the DirectSound microphone.
    audio_device_index: int | None = None

    def __post_init__(self) -> None:
        """Validate application settings."""

        if not self.model_size.strip():
            raise ValueError("Model size cannot be empty.")

        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be cpu or cuda.")

        if self.compute_type not in {
            "int8",
            "float16",
            "float32",
        }:
            raise ValueError("Unsupported compute type.")

        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if not isinstance(self.audio_directory, Path):
            raise TypeError(
                "audio_directory must be a pathlib.Path."
            )

        if (
            self.audio_device_index is not None
            and self.audio_device_index < 0
        ):
            raise ValueError("Audio device index cannot be negative.")

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load configuration from environment variables."""

        raw_device_index = os.getenv("STT_AUDIO_DEVICE_INDEX")

        return cls(
            model_size=os.getenv("STT_MODEL_SIZE", "small.en"),
            device=os.getenv("STT_DEVICE", "cpu"),
            compute_type=os.getenv(
                "STT_COMPUTE_TYPE", "int8"
            ),
            sample_rate=int(
                os.getenv("STT_SAMPLE_RATE", "16000")
            ),
            audio_directory=Path(
                os.getenv("STT_AUDIO_DIRECTORY", "recordings")
            ),
            audio_device_index=(
                int(raw_device_index)
                if raw_device_index
                else None
            ),
        )
