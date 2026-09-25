import os
import torch
import whisper
from faster_whisper import WhisperModel

class WhisperTranscriber:
    def __init__(self, faster=True, language = None, model_size="small", device="cpu"):
        self.faster = faster
        self.language = language
        self.faster_model: WhisperModel | None = None
        self.whisper_model: whisper.Whisper | None = None
        
        if self.faster:
            self.device = device if device in ["cpu", "cuda"] else "cpu"
            self.faster_model = WhisperModel(
                model_size, 
                device=self.device,
                compute_type="int8"
            )
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.whisper_model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path):
        if self.faster:
            return self.transcribe_faster(audio_path)
        else:
            return self.transcribe_whisper(audio_path) 

    def transcribe_faster(self, audio_path, task="transcribe"):
        assert self.faster_model is not None, "faster_model not initialized (did you set faster=True?)"
        segments, info = self.faster_model.transcribe(
            audio_path,
            language=self.language,
            task=task
        )
        output_path = os.path.splitext(audio_path)[0] + "_transcribed.txt" 
        with open(output_path, "w", encoding="utf-8") as f:
            for s in segments:
                line = s.text + " "
                f.write(line)
                
                # progress display
                progress = (s.end / info.duration) * 100
                print(f"Progress: {progress:.1f}%", end="\r", flush=True)
        print(" " * 40, end="\r")  # clear the progress line
        print(f"Transcription completed!\nSaved to {output_path}", end="\r")
                
    def transcribe_whisper(self, audio_path, fp16=False, verbose=False):
        assert self.whisper_model is not None, "whisper_model not initialized (did you set faster=False?)"
        result = self.whisper_model.transcribe(
            audio_path,
            fp16=fp16,
            verbose=verbose
        )
        output_path = os.path.splitext(audio_path)[0] + "_transcribed.txt" 
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(result["text"]))
        print(f"Transcription completed!\t\t\t\nSaved to {output_path}")