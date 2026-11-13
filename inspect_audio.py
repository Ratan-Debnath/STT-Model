
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    audio_path = Path(
        r"C:\Users\Think Pad\PycharmProjects"
        r"\STT_model_IELTS_v1\stt_app\recordings"
        r"\recording_20261109_070517.wav"
    )

    audio, sample_rate = sf.read(audio_path)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    dbfs_rms = 20 * np.log10(max(rms, 1e-10))
    dbfs_peak = 20 * np.log10(max(peak, 1e-10))

    print("\nAUDIO QUALITY REPORT")
    print("--------------------")
    print("Sample rate:", sample_rate)
    print("Duration:", round(len(audio) / sample_rate, 2))
    print("RMS amplitude:", round(float(rms), 6))
    print("Peak amplitude:", round(float(peak), 6))
    print("RMS dBFS:", round(float(dbfs_rms), 2))
    print("Peak dBFS:", round(float(dbfs_peak), 2))
    print("Minimum:", float(np.min(audio)))
    print("Maximum:", float(np.max(audio)))

    silence_threshold = 0.01
    active_ratio = np.mean(np.abs(audio) > silence_threshold)

    print(
        "Active audio percentage:",
        round(float(active_ratio * 100), 2),
        "%"
    )


if __name__ == "__main__":
    main()
