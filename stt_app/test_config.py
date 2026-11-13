from stt_app.config import Settings

settings = Settings.from_environment()

print("Configuration loaded successfully!")
print("Model:", settings.model_size)
print("Device:", settings.device)
print("Compute:", settings.compute_type)
print("Audio directory:", settings.audio_directory)