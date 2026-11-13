# STT Model – Speech-to-Text System

A Python-based Speech-to-Text (STT) application that captures audio from a microphone and converts spoken language into text using an automatic speech recognition (ASR) model.

## Features

- Microphone audio recording
- Audio input processing
- Speech-to-text transcription
- Whisper-based speech recognition
- Configurable application settings
- Modular Python project structure

## Tech Stack

- **Language:** Python
- **Speech Recognition:** Whisper / Faster-Whisper
- **Audio Processing:** Python audio libraries
- **Environment:** Python Virtual Environment

## Project Structure

```text
STT-Model/
│
├── stt_app/
│   ├── __init__.py
│   ├── audio_recorder.py
│   ├── config.py
│   ├── main.py
│   ├── transcriber.py
│   ├── test_config.py
│   └── test_whisper_clean.py
│
├── inspect_audio.py
├── requirements.txt
├── test_audio_diagnostics.py
├── test_directsound_stream.py
├── test_microphone_devices.py
├── test_recording_pipeline.py
├── test_whisper.py
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ratan-Debnath/STT-Model.git
cd STT-Model
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

Run the application from the project root:

```bash
python -m stt_app.main
```

Make sure your microphone is connected and microphone permissions are enabled.

## Testing

Run the available diagnostic and test scripts:

```bash
python test_microphone_devices.py
python test_audio_diagnostics.py
python test_recording_pipeline.py
python test_whisper.py
```

## Configuration

Application settings are managed in:

```text
stt_app/config.py
```

Audio recording and transcription logic are separated into dedicated modules for easier maintenance and development.

## Future Improvements

- Real-time speech recognition
- Improved transcription accuracy
- Noise reduction and audio enhancement
- Multi-language speech recognition
- REST API integration
- Optimized inference performance

## Author

**Ratan Debnath**

AI Engineer | Machine Learning | NLP | Speech Processing

---

*This project is developed for research, experimentation, and practical applications of automatic speech recognition.*
