import math
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import resample_poly


TARGET_RATE = 16000
DURATION = 10
OUTPUT_DIR = Path("diagnostic_recordings")


def find_microphone():
    for index, device in enumerate(sd.query_devices()):
        hostapi = sd.query_hostapis(device["hostapi"])["name"]

        if (
            "Microphone Array" in device["name"]
            and "DirectSound" in hostapi
            and device["max_input_channels"] > 0
        ):
            return index, device

    raise RuntimeError("Intel microphone DirectSound device not found.")


def report_audio(name: str, audio: np.ndarray) -> None:
    normalized = audio.astype(np.float32) / 32768.0

    rms = float(np.sqrt(np.mean(normalized ** 2)))
    peak = float(np.max(np.abs(normalized)))
    dbfs = 20 * np.log10(max(rms, 1e-10))

    print(f"\n{name}")
    print("-" * 40)
    print(f"RMS: {rms:.6f}")
    print(f"Peak: {peak:.6f}")
    print(f"RMS dBFS: {dbfs:.2f}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device_index, device = find_microphone()

    native_rate = int(device["default_samplerate"])

    print("Selected microphone:", device["name"])
    print("Host API:", sd.query_hostapis(device["hostapi"])["name"])
    print("Device index:", device_index)
    print("Native sample rate:", native_rate)

    input(
        "\nSpeak continuously for 10 seconds. "
        "Press ENTER when ready..."
    )

    print("\nRECORDING NOW — SPEAK CLEARLY!")

    audio = sd.rec(
        frames=DURATION * native_rate,
        samplerate=native_rate,
        channels=1,
        dtype="int16",
        device=device_index,
        blocking=True,
    ).reshape(-1)

    if audio.size == 0:
        raise RuntimeError("Recording is empty.")

    report_audio("NATIVE RECORDING", audio)

    divisor = math.gcd(native_rate, TARGET_RATE)

    resampled = resample_poly(
        audio.astype(np.float32),
        up=TARGET_RATE // divisor,
        down=native_rate // divisor,
    )

    resampled = np.clip(
        np.rint(resampled),
        -32768,
        32767,
    ).astype(np.int16)

    report_audio("RESAMPLED RECORDING", resampled)

    native_path = OUTPUT_DIR / "microphone_native.wav"
    resampled_path = OUTPUT_DIR / "microphone_16000hz.wav"

    write(str(native_path), native_rate, audio)
    write(str(resampled_path), TARGET_RATE, resampled)

    print("\nFILES SAVED")
    print(native_path.resolve())
    print(resampled_path.resolve())

    print("\nRecording pipeline test completed.")


if __name__ == "__main__":
    main()