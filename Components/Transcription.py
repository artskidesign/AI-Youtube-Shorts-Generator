from faster_whisper import WhisperModel
import torch
from moviepy.editor import VideoFileClip
import os

class Transcription:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = WhisperModel("base.en", device=self.device)
        print("Whisper model loaded")

    def extract_audio(self, video_path):
        """Extract audio from video file"""
        try:
            video = VideoFileClip(video_path)
            audio_path = "temp_audio.wav"
            video.audio.write_audiofile(audio_path)
            video.close()
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def transcribe_video(self, video_path):
        """Transcribe video by first extracting audio then transcribing"""
        try:
            # Extract audio from video
            audio_path = self.extract_audio(video_path)
            if not audio_path:
                return []

            # Transcribe audio
            print("Transcribing audio...")
            segments, info = self.model.transcribe(
                audio=audio_path,
                beam_size=5,
                language="en",
                max_new_tokens=128,
                condition_on_previous_text=False
            )
            
            # Convert segments to list and extract text with timestamps
            segments = list(segments)
            extracted_texts = [[segment.text, segment.start, segment.end] for segment in segments]

            # Clean up temporary audio file
            os.remove(audio_path)
            
            return extracted_texts

        except Exception as e:
            print(f"Transcription Error: {e}")
            return []

    def get_full_transcript(self, transcriptions):
        """Convert transcription segments into a single string"""
        transcript = ""
        for text, start, end in transcriptions:
            transcript += f"{start:.2f} - {end:.2f}: {text}\n"
        return transcript

if __name__ == "__main__":
    transcriber = Transcription()
    video_path = input("Enter video path: ")
    transcriptions = transcriber.transcribe_video(video_path)
    print("\nFull Transcript:")
    print(transcriber.get_full_transcript(transcriptions))