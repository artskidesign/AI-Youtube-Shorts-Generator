import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import os
from collections import deque

class BallTracker:
    def __init__(self):
        print("Initializing ball tracker using OpenCV")
        # Parameters for circle detection
        self.min_radius = 10
        self.max_radius = 100
        self.min_dist = 50
        self.param1 = 50  # for edge detection
        self.param2 = 30  # threshold for center detection
        
        # Create output directory if it doesn't exist
        self.output_dir = "tracked_videos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Target dimensions for 9:16 ratio (1080x1920)
        self.target_width = 1080
        self.target_height = 1920
        
        # Motion smoothing parameters
        self.smooth_window = 30  # Number of frames to average
        self.position_history = deque(maxlen=self.smooth_window)
        self.last_crop_pos = None

    def smooth_position(self, current_pos):
        """Smooth the camera movement using moving average"""
        if current_pos is None and self.last_crop_pos is not None:
            return self.last_crop_pos
        
        if current_pos is None:
            return None
            
        x, y = current_pos
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            self.last_crop_pos = (x, y)
            return (x, y)
        
        # Calculate smoothed position
        positions = list(self.position_history)
        smooth_x = int(sum(p[0] for p in positions) / len(positions))
        smooth_y = int(sum(p[1] for p in positions) / len(positions))
        
        # Limit the movement speed
        if self.last_crop_pos is not None:
            last_x, last_y = self.last_crop_pos
            max_move = 30  # Maximum pixels to move per frame
            
            dx = smooth_x - last_x
            dy = smooth_y - last_y
            
            # Apply speed limit
            if abs(dx) > max_move:
                smooth_x = last_x + (max_move if dx > 0 else -max_move)
            if abs(dy) > max_move:
                smooth_y = last_y + (max_move if dy > 0 else -max_move)
        
        self.last_crop_pos = (smooth_x, smooth_y)
        return (smooth_x, smooth_y)

    def convert_to_vertical(self, frame, ball_coords=None):
        """Convert frame to vertical 9:16 format, focusing on the ball if detected"""
        h, w = frame.shape[:2]
        
        # Calculate crop window
        crop_width = int(w * 0.4)  # Crop to 40% of original width
        crop_height = int(crop_width * (16/9))  # Maintain 9:16 ratio
        
        if ball_coords:
            x, y, _, _ = ball_coords
            smooth_pos = self.smooth_position((x, y))
            if smooth_pos:
                x, y = smooth_pos
        else:
            if self.last_crop_pos is not None:
                x, y = self.last_crop_pos
            else:
                x, y = w//2, h//2
                self.last_crop_pos = (x, y)
        
        # Calculate crop coordinates with smooth position
        x_start = max(0, x - crop_width//2)
        x_end = min(w, x + crop_width//2)
        y_start = max(0, y - crop_height//2)
        y_end = min(h, y + crop_height//2)
        
        # Adjust if crop window goes out of bounds
        if x_start < 0:
            x_end = min(w, crop_width)
            x_start = 0
        if x_end > w:
            x_start = max(0, w - crop_width)
            x_end = w
        if y_start < 0:
            y_end = min(h, crop_height)
            y_start = 0
        if y_end > h:
            y_start = max(0, h - crop_height)
            y_end = h
        
        # Ensure we have the correct crop size
        actual_width = x_end - x_start
        actual_height = y_end - y_start
        
        if actual_width != crop_width or actual_height != crop_height:
            # Adjust crop window to maintain size
            if actual_width != crop_width:
                diff = crop_width - actual_width
                if x_start == 0:
                    x_end = min(w, x_end + diff)
                else:
                    x_start = max(0, x_start - diff)
            
            if actual_height != crop_height:
                diff = crop_height - actual_height
                if y_start == 0:
                    y_end = min(h, y_end + diff)
                else:
                    y_start = max(0, y_start - diff)
        
        # Crop the frame
        cropped = frame[y_start:y_end, x_start:x_end]
        
        # Resize to target dimensions
        vertical = cv2.resize(cropped, (self.target_width, self.target_height))
        return vertical

    def detect_ball(self, frame):
        """Detect circular objects (balls) in a frame using OpenCV"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=self.min_dist,
                param1=self.param1,
                param2=self.param2,
                minRadius=self.min_radius,
                maxRadius=self.max_radius
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Get the most prominent circle (usually the ball)
                x, y, r = circles[0][0]
                return (int(x), int(y), int(r * 2), int(r * 2))
                
        except Exception as e:
            print(f"Error during ball detection: {e}")
        
        return None

    def process_video(self, input_path, output_name=None):
        """Process video file and track ball movement"""
        print(f"Processing video: {input_path}")
        try:
            # Generate output path
            if output_name is None:
                input_filename = os.path.basename(input_path)
                output_name = f"tracked_{input_filename}"
            output_path = os.path.join(self.output_dir, output_name)
            
            print(f"Output will be saved to: {output_path}")
            
            # Reset motion smoothing
            self.position_history.clear()
            self.last_crop_pos = None
            
            # Load video
            clip = VideoFileClip(input_path)
            frames = []
            total_frames = int(clip.fps * clip.duration)
            processed_frames = 0
            
            # Process each frame
            for frame in clip.iter_frames():
                # Convert frame to BGR (OpenCV format)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Detect ball
                ball_coords = self.detect_ball(frame_bgr)
                
                # Convert to vertical format with ball tracking
                frame_vertical = self.convert_to_vertical(frame_bgr, ball_coords)
                
                if ball_coords:
                    x, y, w, h = ball_coords
                    # Draw circle around ball (adjusted for cropped frame)
                    cv2.circle(frame_vertical, (self.target_width//2, self.target_height//2), 
                             w//2, (0, 255, 0), 2)
                    # Add a center point
                    cv2.circle(frame_vertical, (self.target_width//2, self.target_height//2), 
                             2, (0, 0, 255), 3)
                
                # Convert back to RGB for moviepy
                frame_rgb = cv2.cvtColor(frame_vertical, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Update progress
                processed_frames += 1
                if processed_frames % 10 == 0:  # Update every 10 frames
                    print(f"Processing frames: {processed_frames}/{total_frames} ({(processed_frames/total_frames)*100:.1f}%)")
            
            print("Creating output video...")
            
            # Create a clip from processed frames
            processed_clip = ImageSequenceClip(frames, fps=clip.fps)
            
            # Add audio from original clip
            processed_clip = processed_clip.set_audio(clip.audio)
            
            # Write the final video
            processed_clip.write_videofile(
                output_path,
                fps=clip.fps,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            clip.close()
            processed_clip.close()
            
            print(f"Video processing complete. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def get_ball_coordinates(self, video_path):
        """Extract ball coordinates throughout the video"""
        coordinates = []
        try:
            clip = VideoFileClip(video_path)
            total_frames = int(clip.fps * clip.duration)
            processed_frames = 0
            
            print("Extracting ball coordinates...")
            for frame in clip.iter_frames():
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ball_coords = self.detect_ball(frame_bgr)
                if ball_coords:
                    x, y, w, h = ball_coords
                    coordinates.append((x, y))
                else:
                    coordinates.append(None)
                    
                # Update progress
                processed_frames += 1
                if processed_frames % 10 == 0:  # Update every 10 frames
                    print(f"Processing frames: {processed_frames}/{total_frames} ({(processed_frames/total_frames)*100:.1f}%)")
            
            clip.close()
            print("Ball coordinate extraction complete")
            
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            
        return coordinates
