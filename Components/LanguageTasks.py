from dotenv import load_dotenv
import os
import openai
import json

load_dotenv()

class LanguageTasks:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API')
        if not self.openai_key:
            raise ValueError("OPENAI_API key not found in environment variables")
        openai.api_key = self.openai_key
        
        self.system_prompt = '''
        Based on the Transcription provided with start and end times, highlight the main parts in less than 1 min which can be directly converted into a short. 
        Highlight it such that it's interesting and also keep the timestamps for the clip to start and end. Only select a continuous part of the video.

        Follow this Format and return in valid json:
        [{
        "start": "Start time of the clip",
        "content": "Highlight Text",
        "end": "End Time for the highlighted clip"
        }]

        It should be one continuous clip as it will then be cut from the video and uploaded as a short video.
        So only have one start, end and content.

        Return ONLY proper JSON, no explanation or other text.
        '''

    def extract_times(self, json_string):
        """Extract start and end times from JSON response"""
        try:
            data = json.loads(json_string)
            start_time = float(data[0]["start"])
            end_time = float(data[0]["end"])
            return int(start_time), int(end_time)
        except Exception as e:
            print(f"Error extracting times: {e}")
            return 0, 0

    def get_highlights_from_transcription(self, transcriptions):
        """Get highlights from video transcription using OpenAI"""
        try:
            # Convert transcriptions to a readable format
            transcript_text = ""
            for text, start, end in transcriptions:
                transcript_text += f"{start:.2f} - {end:.2f}: {text}\n"

            # Get response from OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": transcript_text}
                ]
            )

            # Extract the JSON response
            json_response = response.choices[0].message.content
            
            # Parse and validate the response
            highlights = json.loads(json_response)
            if not highlights or not isinstance(highlights, list):
                raise ValueError("Invalid response format")

            return highlights[0]  # Return the first (and should be only) highlight

        except Exception as e:
            print(f"Error getting highlights: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    language_tasks = LanguageTasks()
    example_transcription = [
        ["This is a test", 0.0, 5.0],
        ["Another test segment", 5.0, 10.0]
    ]
    highlights = language_tasks.get_highlights_from_transcription(example_transcription)
    print(json.dumps(highlights, indent=2))
