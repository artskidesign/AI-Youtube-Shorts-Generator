import os
import re
from pytubefix import YouTube

class YoutubeDownloader:
    def __init__(self):
        self.output_dir = "videos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def sanitize_filename(self, filename):
        """Remove special characters and spaces from filename"""
        # Replace special characters and spaces with underscores
        sanitized = re.sub(r'[^\w\-_.]', '_', filename)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized

    def download_video(self, url):
        try:
            yt = YouTube(url)
            print(f"Downloading video: {yt.title}")

            # Create sanitized base filename
            base_filename = self.sanitize_filename(yt.title)
            
            # Get available streams
            streams = yt.streams.filter(progressive=True).order_by('resolution').desc()
            print("Available video streams:")
            for i, stream in enumerate(streams):
                print(f"{i}. Resolution: {stream.resolution}, Size: {stream.filesize_mb:.2f} MB, Type: Progressive")

            # Get user choice
            choice = int(input("Enter the number of the video stream to download: "))
            selected_stream = streams[choice]

            # Set up output paths
            video_filename = f"{base_filename}.mp4"
            output_path = os.path.join(self.output_dir, video_filename)

            # Download the video
            print(f"Downloading video file...")
            selected_stream.download(output_path=self.output_dir, filename=video_filename)
            print(f"Video downloaded successfully to: {output_path}")
            
            return output_path

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please make sure you have the latest version of pytube installed.")
            print("You can update it by running:")
            print("pip install --upgrade pytubefix")
            return None

if __name__ == "__main__":
    downloader = YoutubeDownloader()
    youtube_url = input("Enter YouTube video URL: ")
    downloader.download_video(youtube_url)
