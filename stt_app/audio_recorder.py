
"""Microphone recording and audio preprocessing."""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly

from stt_app.config import Settings


logger = logging.getLogger(__name__)

MIN_RMS_AMPLITUDE = 0.001
MIN_PEAK_AMPLITUDE = 0.005


class MicrophoneRecorder:
    """Record microphone audio and save standardized WAV files."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the microphone recorder."""

        self.settings = settings

        self.settings.audio_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _find_input_device(self) -> tuple[int, dict]:
        """Find the configured or preferred DirectSound microphone."""

        if self.settings.audio_device_index is not None:
            index = self.settings.audio_device_index

            try:
                device = sd.query_devices(index)
            except Exception as exc:
                raise RuntimeError(
                    f"Audio device {index} is unavailable."
                ) from exc

            if device["max_input_channels"] < 1:
                raise RuntimeError(
                    f"Device {index} has no input channels."
                )

            return index, device

        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        for index, device in enumerate(devices):
            hostapi_name = hostapis[device["hostapi"]]["name"]

            is_directsound = (
                "directsound" in hostapi_name.lower()
            )

            is_microphone_array = (
                "microphone array" in device["name"].lower()
            )

            has_input = device["max_input_channels"] > 0

            if (
                is_directsound
                and is_microphone_array
                and has_input
            ):
                logger.info(
                    "Selected microphone: %s (device %s)",
                    device["name"],
                    index,
                )

                return index, device

        raise RuntimeError(
            "Intel microphone array was not found through "
            "Windows DirectSound. Check your microphone device."
        )

    def record(self) -> Path:
        """Record microphone audio until the user presses Enter."""

        try:
            device_index, device = self._find_input_device()

            native_rate = int(device["default_samplerate"])

            if device["max_input_channels"] < 1:
                raise RuntimeError(
                    "No microphone input channel available."
                )

            chunks: list[np.ndarray] = []

            def audio_callback(
                audio: np.ndarray,
                frames: int,
                time_info: object,
                status: sd.CallbackFlags,
            ) -> None:
                """Collect microphone audio chunks."""

                if status:
                    logger.warning(
                        "Audio recording status: %s",
                        status,
                    )

                chunks.append(audio.copy())

            logger.info(
                "Starting recording | Device: %s | Rate: %s Hz",
                device["name"],
                native_rate,
            )

            print("\nRecording started...")
            print(f"Microphone: {device['name']}")
            print(f"Sample rate: {native_rate} Hz")
            print("Speak English into your microphone.")
            print("Press ENTER when you finish speaking.\n")

            with sd.InputStream(
                device=device_index,
                samplerate=native_rate,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            ):
                input("Recording... Press ENTER to stop: ")

            if not chunks:
                raise ValueError("No audio was recorded.")

            audio = np.concatenate(
                chunks,
                axis=0,
            ).reshape(-1)

            if audio.size == 0:
                raise ValueError("Recording is empty.")

            # Validate the original microphone signal.
            normalized = audio.astype(np.float32) / 32768.0

            rms = float(
                np.sqrt(np.mean(normalized ** 2))
            )

            peak = float(
                np.max(np.abs(normalized))
            )

            rms_dbfs = 20 * np.log10(
                max(rms, 1e-10)
            )

            logger.info(
                "Audio quality | RMS: %.6f | Peak: %.6f | "
                "RMS dBFS: %.2f",
                rms,
                peak,
                rms_dbfs,
            )

            print("\nAudio quality report")
            print("--------------------")
            print(f"RMS: {rms:.6f}")
            print(f"Peak: {peak:.6f}")
            print(f"RMS dBFS: {rms_dbfs:.2f}")

            if (
                rms < MIN_RMS_AMPLITUDE
                or peak < MIN_PEAK_AMPLITUDE
            ):
                raise ValueError(
                    "Recording contains insufficient audio. "
                    "Check microphone volume and speak clearly."
                )

            # Convert native sample rate to target sample rate.
            processed_audio = self._resample(
                audio=audio,
                original_rate=native_rate,
                target_rate=self.settings.sample_rate,
            )

            filename = datetime.now(
                timezone.utc
            ).strftime(
                "recording_%Y%m%d_%H%M%S_%f.wav"
            )

            output_path = (
                self.settings.audio_directory / filename
            )

            write(
                str(output_path),
                self.settings.sample_rate,
                processed_audio,
            )

            logger.info(
                "Audio saved successfully: %s",
                output_path,
            )

            print(f"\nAudio saved: {output_path}")

            return output_path

        except (ValueError, RuntimeError):
            raise

        except Exception as exc:
            logger.exception(
                "Microphone recording failed."
            )

            raise RuntimeError(
                "Unable to record microphone audio."
            ) from exc

    @staticmethod
    def _resample(
        audio: np.ndarray,
        original_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        """Resample audio to the required sample rate."""

        if original_rate <= 0 or target_rate <= 0:
            raise ValueError(
                "Sample rates must be positive."
            )

        if original_rate == target_rate:
            return audio.astype(np.int16, copy=False)

        divisor = math.gcd(
            original_rate,
            target_rate,
        )

        resampled = resample_poly(
            audio.astype(np.float32),
            up=target_rate // divisor,
            down=original_rate // divisor,
        )

        return np.clip(
            np.rint(resampled),
            -32768,
            32767,
        ).astype(np.int16)
