from Components.BallTracker import BallTracker
from Components.YoutubeDownloader import YoutubeDownloader
from Components.Transcription import Transcription
from Components.LanguageTasks import LanguageTasks
import os

def main():
    # Initialize components
    downloader = YoutubeDownloader()
    ball_tracker = BallTracker()
    transcriber = Transcription()
    language_tasks = LanguageTasks()

    # Get YouTube video URL from user
    video_url = input("Enter YouTube video URL: ")
    
    # Download video
    print("Downloading video...")
    video_path = downloader.download_video(video_url)
    
    if video_path is None or not os.path.exists(video_path):
        print("Failed to download video. Please try again.")
        return

    # Process video with ball tracking
    print("Processing video and tracking ball...")
    processed_video = ball_tracker.process_video(video_path)
    
    if processed_video is None:
        print("Failed to process video. Please try again.")
        return

    # Get ball movement coordinates
    print("Extracting ball coordinates...")
    coordinates = ball_tracker.get_ball_coordinates(processed_video)
    
    # Get video transcription
    print("Transcribing video...")
    transcription = transcriber.transcribe_video(video_path)
    
    # Get highlights using OpenAI
    print("Generating highlights...")
    if transcription:
        highlights = language_tasks.get_highlights_from_transcription(transcription)
        if highlights:
            print("\nHighlights generated successfully!")
            print(highlights)
    
    print("\nProcessing complete!")
    if processed_video:
        print(f"Processed video saved as: {processed_video}")
    print("Ball movement coordinates extracted")
    if coordinates:
        print(f"Number of frames with ball detected: {sum(1 for coord in coordinates if coord is not None)}")

if __name__ == "__main__":
    main()
