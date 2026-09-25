# Whisper Transcriber (Audio to Text)

A small Python wrapper for transcribing audio files using either [OpenAI Whisper](https://github.com/openai/whisper) or [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with a single unified interface.

## Features

- Switch between two transcription backends:
  - **faster-whisper** — CPU-friendly, quantized (`int8`), generally faster on CPU
  - **openai-whisper** — supports Apple Silicon GPU acceleration via MPS
- Automatically writes the transcript to a `.txt` file next to the source audio
- Live progress display while transcribing (faster-whisper backend)

## Requirements

- Python 3.9+

## Installation

```bash
python3 -m venv whisper-env
source whisper-env/bin/activate

pip install git+https://github.com/openai/whisper.git
pip install faster-whisper
pip install torch
```

## Usage

```python
from transcribe import WhisperTranscriber

audio_path = "your/audio/path/example.mp3"

transcriber = WhisperTranscriber()
transcriber.transcribe(audio_path)
```

This produces `your/audio/path/example_transcribed.txt` containing the transcript.

### Choosing a backend

```python
# faster-whisper (default) — good for CPU
transcriber = WhisperTranscriber(faster=True, model_size="small", device="cpu")

# openai-whisper — good for Apple Silicon (MPS) or CUDA
transcriber = WhisperTranscriber(faster=False, model_size="medium")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `faster` | `True` | Use `faster-whisper` if `True`, otherwise `openai-whisper` |
| `language` | `None` | Force a language code (e.g. `"en"`, `"pt"`); auto-detected if `None`. Only used by the `faster-whisper` backend |
| `model_size` | `"small"` | Model size (`tiny`, `base`, `small`, `medium`, `large`, ...) |
| `device` | `"cpu"` | Device for `faster-whisper` (`"cpu"` or `"cuda"`). The `openai-whisper` backend auto-selects `"mps"` on Apple Silicon if available, otherwise `"cpu"` |

## Project structure

```
.
├── transcribe.py       # WhisperTranscriber class
├── usecase.ipynb        # example notebook
└── audios/
    ├── example.mp3
    └── example_transcribed.txt
```

## Notes

- `faster-whisper` does not support Apple's MPS backend — only `"cpu"` and `"cuda"`. The `openai-whisper` backend is the one that benefits from MPS on Mac.
- Output files are always saved as `<original_filename>_transcribed.txt` in the same folder as the source audio.