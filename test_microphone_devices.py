
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 44100
DURATION = 5

DEVICES = {
    "MME": 1,
    "DirectSound": 6,
    "WASAPI": 12,
}


def test_microphone(name: str, device_index: int) -> None:
    print("\n" + "=" * 55)
    print(f"Testing {name} | Device index: {device_index}")
    print("=" * 55)

    try:
        device = sd.query_devices(device_index)
        print("Device:", device["name"])
        print("Default sample rate:", device["default_samplerate"])
        print("Input channels:", device["max_input_channels"])

        sd.check_input_settings(
            device=device_index,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
        )

        input(
            "\nSpeak continuously for 5 seconds. "
            "Press ENTER to start recording..."
        )

        print("RECORDING... SPEAK NOW!")

        audio = sd.rec(
            frames=int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_index,
        )

        sd.wait()

        audio = audio.reshape(-1)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        dbfs = 20 * np.log10(max(rms, 1e-10))
        active = float(np.mean(np.abs(audio) > 0.01) * 100)

        print("\nRESULT")
        print("RMS amplitude:", round(rms, 6))
        print("Peak amplitude:", round(peak, 6))
        print("RMS dBFS:", round(dbfs, 2))
        print("Active audio percentage:", round(active, 2), "%")

        if peak < 0.001:
            print("WARNING: Extremely weak microphone signal.")
        else:
            print("Microphone signal detected.")

    except Exception as exc:
        print("DEVICE TEST FAILED:", repr(exc))


def main() -> None:
    for name, device_index in DEVICES.items():
        test_microphone(name, device_index)

    print("\nALL MICROPHONE TESTS COMPLETED.")


if __name__ == "__main__":
    main()
