from src.insanely_fast_whisper.transcribe import transcribe
from src.segment import segment_text

class AudioTranscriber:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.transcription = ""
        self.segments = []

    def perform_transcription(self):
        self.transcription = transcribe(self.audio_file)

    def segment_transcription(self):
        if self.transcription:
            self.segments = segment_text(self.transcription)

    def get_segments(self):
        return self.segments


# Example usage of the classes
audio_transcriber = AudioTranscriber("path/to/audio/file.mp3")
audio_transcriber.perform_transcription()
audio_transcriber.segment_transcription()

segments = audio_transcriber.get_segments()
