import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


DEVICE_INDEX = 8
SAMPLE_RATE = 44100
DURATION = 10


def main() -> None:
    chunks = []

    def audio_callback(
        audio,
        frames,
        time_info,
        status,
    ):
        if status:
            print("Recording status:", status)

        chunks.append(audio.copy())

    print("Selected device:")
    print(sd.query_devices(DEVICE_INDEX))

    input("\nPress ENTER, then speak for 10 seconds...")

    print("\nRECORDING...")

    with sd.InputStream(
        device=DEVICE_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=audio_callback,
    ):
        input("Recording... Press ENTER to stop: ")

    if not chunks:
        raise RuntimeError("No audio chunks captured.")

    audio = np.concatenate(chunks, axis=0).reshape(-1)

    normalized = audio.astype(np.float32) / 32768.0

    rms = float(np.sqrt(np.mean(normalized ** 2)))
    peak = float(np.max(np.abs(normalized)))

    print("\nRESULT")
    print("Samples:", audio.size)
    print("Duration:", audio.size / SAMPLE_RATE)
    print("RMS:", rms)
    print("Peak:", peak)
    print("RMS dBFS:", 20 * np.log10(max(rms, 1e-10)))

    output_path = "directsound_callback_test.wav"

    write(output_path, SAMPLE_RATE, audio)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()