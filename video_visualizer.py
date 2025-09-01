import numpy as np
import librosa
import cv2
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import argparse
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

class VideoMusicVisualizer:
    def __init__(self, width=1920, height=1080, fps=50, text_line1=None, text_line2=None):
        self.width = width
        self.height = height
        self.fps = fps
        
        # Text overlay parameters
        self.text_line1 = text_line1
        self.text_line2 = text_line2
        self.font = None  # Cache font to avoid loading every frame

        # Tweakable parameters for bar behavior
        self.INERTIA = 0.85  # How much inertia bars have (0=no inertia, 0.99=maximum inertia)
        self.SENSITIVITY = 0.5  # How sensitive bars are to changes (0=not sensitive, 1=maximum sensitivity)

        # Visualizer parameters with 5% padding
        self.padding = int(width * 0.05)  # 5% padding on each side
        self.usable_width = width - (2 * self.padding)
        self.num_bars = 30
        self.bar_width = self.usable_width // self.num_bars
        self.max_bar_height = height - 100
        self.min_bar_height = 10

        # Current bar heights for smooth transitions
        self.current_bar_heights = np.zeros(self.num_bars)

        # Audio data
        self.audio_data = None
        self.sample_rate = 44100
        self.duration = 0
        
        # Streaming parameters
        self.window_size = 2048 * 4
        self.hop_length = 512 * 4
        self.max_freq = 8000

        # Colors for bars
        self.bar_color = (255, 255, 255)  # White bars

        # Background
        self.bg_color = (0, 0, 0)  # Black background

    def _generate_gradient_colors(self):
        """Generate colors that match the gradient in your image"""
        colors = []
        for i in range(self.num_bars):
            # Create gradient from magenta -> purple -> blue -> cyan -> red -> orange
            progress = i / (self.num_bars - 1)

            if progress < 0.2:  # Magenta to purple
                r = int(255 * (1 - progress * 2.5))
                g = int(100 * progress * 5)
                b = int(255 * (0.6 + progress * 2))
            elif progress < 0.4:  # Purple to blue
                local_progress = (progress - 0.2) / 0.2
                r = int(150 * (1 - local_progress))
                g = int(100 + 55 * local_progress)
                b = 255
            elif progress < 0.6:  # Blue to cyan
                local_progress = (progress - 0.4) / 0.2
                r = int(50 * local_progress)
                g = int(155 + 100 * local_progress)
                b = 255
            elif progress < 0.8:  # Cyan to red
                local_progress = (progress - 0.6) / 0.2
                r = int(50 + 205 * local_progress)
                g = int(255 * (1 - local_progress * 0.8))
                b = int(255 * (1 - local_progress))
            else:  # Red to orange
                local_progress = (progress - 0.8) / 0.2
                r = 255
                g = int(50 + 155 * local_progress)
                b = int(50 * (1 - local_progress))

            colors.append((r, g, b))
        return colors

    def load_audio(self, file_path):
        """Load audio file without pre-computing spectrum"""
        try:
            print("Loading audio file...")
            # Load audio with librosa
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=44100)
            self.duration = len(self.audio_data) / self.sample_rate

            print(f"Audio loaded: {len(self.audio_data)} samples, {self.sample_rate} Hz, {self.duration:.2f}s")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False

    def _compute_spectrum_for_frame(self, frame_idx, stft_magnitude, freq_indices, hop_length):
        """Compute spectrum for a single frame"""
        # Calculate corresponding audio frame
        time_pos = frame_idx / self.fps
        audio_frame_idx = int(time_pos * self.sample_rate / hop_length)

        if audio_frame_idx < stft_magnitude.shape[1]:
            frame_spectrum = stft_magnitude[freq_indices, audio_frame_idx]
        else:
            frame_spectrum = np.zeros(len(freq_indices))

        # Resample to num_bars
        if len(frame_spectrum) > self.num_bars:
            # Average bins together
            bins_per_bar = len(frame_spectrum) // self.num_bars
            resampled = []
            for i in range(self.num_bars):
                start_idx = i * bins_per_bar
                end_idx = start_idx + bins_per_bar
                if end_idx > len(frame_spectrum):
                    end_idx = len(frame_spectrum)
                avg_magnitude = np.mean(frame_spectrum[start_idx:end_idx])
                resampled.append(avg_magnitude)
        else:
            # Interpolate if we have fewer bins than bars
            resampled = np.interp(
                np.linspace(0, len(frame_spectrum) - 1, self.num_bars),
                np.arange(len(frame_spectrum)),
                frame_spectrum
            )

        return np.array(resampled)

    def create_frame(self, spectrum):
        """Create a single frame of the visualization"""
        # Create PIL image for easier drawing
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

        # Draw text lines if provided
        if self.text_line1 or self.text_line2:
            try:
                # Load font once and cache it
                if self.font is None:
                    font_size = int(self.height * 0.10)  # 10% of video height
                    print(f"Debug: Calculated font size: {font_size} for height: {self.height}")
                    
                    try:
                        # Common system fonts
                        font_paths = [
                            "/System/Library/Fonts/Helvetica.ttc",  # macOS
                            "/System/Library/Fonts/Arial.ttf",  # macOS fallback
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                            "C:/Windows/Fonts/arial.ttf"  # Windows
                        ]
                        self.font = None
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                self.font = ImageFont.truetype(font_path, font_size)
                                print(f"Debug: Loaded font from {font_path} with size {font_size}")
                                break
                        if self.font is None:
                            # Try to create a scalable bitmap font as fallback
                            try:
                                self.font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
                                print(f"Debug: Using Menlo font with size {font_size}")
                            except:
                                self.font = ImageFont.load_default()
                                print(f"Debug: Using default font (size not adjustable)")
                    except Exception as e:
                        print(f"Debug: Font loading failed: {e}")
                        self.font = ImageFont.load_default()
                
                font_size = int(self.height * 0.10)  # Recalculate for spacing
                text_y_start = int(self.height * 0.04)  # 4% from top
                line_spacing = int(font_size * 1.2)  # 120% of font size
                
                if self.text_line1:
                    bbox = draw.textbbox((0, 0), self.text_line1, font=self.font)
                    text_width = bbox[2] - bbox[0]
                    text_x = (self.width - text_width) // 2
                    draw.text((text_x, text_y_start), self.text_line1, fill=(255, 255, 255), font=self.font)
                
                if self.text_line2:
                    bbox = draw.textbbox((0, 0), self.text_line2, font=self.font)
                    text_width = bbox[2] - bbox[0]
                    text_x = (self.width - text_width) // 2
                    text_y = text_y_start + line_spacing
                    draw.text((text_x, text_y), self.text_line2, fill=(255, 255, 255), font=self.font)
                    
            except Exception as e:
                print(f"Warning: Could not render text: {e}")

        for i, magnitude in enumerate(spectrum):
            # Apply sensitivity to the input magnitude
            adjusted_magnitude = magnitude * self.SENSITIVITY
            
            # Calculate target bar height
            target_bar_height = self.min_bar_height + adjusted_magnitude * self.max_bar_height
            
            # Apply inertia - smoothly transition current height toward target
            self.current_bar_heights[i] = (
                self.current_bar_heights[i] * self.INERTIA +
                target_bar_height * (1 - self.INERTIA)
            )

            bar_height = int(self.current_bar_heights[i])

            # Bar position with padding
            x = self.padding + i * self.bar_width + self.bar_width // 4
            y = self.height - bar_height - 50
            width = max(2, self.bar_width // 2)  # Ensure minimum width

            # Draw main bar
            draw.rectangle([x, y, x + width, y + bar_height], fill=self.bar_color)

        # Convert to numpy array for OpenCV
        return np.array(img)

    def render_video(self, output_path, audio_path=None, progress_callback=None):
        """Render video with parallel frequency analysis and frame generation"""
        if self.audio_data is None:
            print("No audio data loaded!")
            return False

        total_frames = int(self.duration * self.fps)
        print(f"Rendering video: {output_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {total_frames}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, float(self.fps), (self.width, self.height))

        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            return False

        try:
            # Compute STFT once for the entire audio
            print("Computing STFT...")
            stft = librosa.stft(self.audio_data, n_fft=self.window_size, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            db_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # Get frequency indices for filtering
            freq_bins = np.linspace(0, self.sample_rate // 2, self.window_size // 2 + 1)
            freq_indices = np.where(freq_bins <= self.max_freq)[0]
            
            # Normalize spectrum data
            min_val = np.min(db_magnitude[freq_indices])
            max_val = np.max(db_magnitude[freq_indices])
            if max_val > min_val:
                db_magnitude = (db_magnitude - min_val) / (max_val - min_val)
            
            print("Starting parallel frame generation...")
            
            # Frame buffer and processing queues
            frame_queue = Queue(maxsize=50)  # Limit memory usage
            spectrum_queue = Queue(maxsize=50)
            
            def spectrum_worker():
                """Worker to compute spectrum data"""
                for frame_idx in range(total_frames):
                    spectrum = self._compute_spectrum_for_frame(
                        frame_idx, db_magnitude, freq_indices, self.hop_length
                    )
                    # Apply power scaling for better visualization
                    spectrum = np.power(spectrum, 0.7)
                    spectrum_queue.put((frame_idx, spectrum))
                spectrum_queue.put(None)  # Signal end
            
            def frame_worker():
                """Worker to generate video frames"""
                while True:
                    item = spectrum_queue.get()
                    if item is None:
                        frame_queue.put(None)
                        break
                    
                    frame_idx, spectrum = item
                    frame_rgb = self.create_frame(spectrum)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    frame_queue.put((frame_idx, frame_bgr))
            
            # Start worker threads
            spectrum_thread = threading.Thread(target=spectrum_worker)
            frame_thread = threading.Thread(target=frame_worker)
            
            spectrum_thread.start()
            frame_thread.start()
            
            # Write frames as they become available
            frames_written = 0
            while frames_written < total_frames:
                try:
                    item = frame_queue.get(timeout=30)
                    if item is None:
                        break
                    
                    frame_idx, frame_bgr = item
                    video_writer.write(frame_bgr)
                    frames_written += 1
                    
                    # Progress reporting
                    if progress_callback and frames_written % 10 == 0:
                        progress = (frames_written / total_frames) * 70 + 20  # 20-90% range
                        progress_callback(progress)
                    elif frames_written % (total_frames // 10 + 1) == 0:
                        progress = (frames_written / total_frames) * 100
                        print(f"Progress: {progress:.1f}%")
                        
                except Empty:
                    print("Timeout waiting for frame")
                    break
            
            # Wait for threads to finish
            spectrum_thread.join()
            frame_thread.join()
            
            print("Video rendering complete!")

        except Exception as e:
            print(f"Error during rendering: {e}")
            return False
        finally:
            video_writer.release()

        # If audio path provided, add audio to video using ffmpeg
        if audio_path and os.path.exists(audio_path):
            print("Adding audio to video...")
            temp_video = output_path.replace('.mp4', '_temp.mp4')
            os.rename(output_path, temp_video)

            # Use ffmpeg to combine video and audio
            cmd = f'ffmpeg -i "{temp_video}" -i "{audio_path}" -c:v copy -c:a aac -shortest -y "{output_path}"'
            result = os.system(cmd)

            if result == 0:
                os.remove(temp_video)
                print("Audio added successfully!")
                if progress_callback:
                    progress_callback(100)
            else:
                print("Warning: Could not add audio. Make sure ffmpeg is installed.")
                os.rename(temp_video, output_path)

        return True

def main():
    parser = argparse.ArgumentParser(description='Generate music visualization video')
    parser.add_argument('input_audio', help='Input audio file path')
    parser.add_argument('-o', '--output', default='visualization.mp4', help='Output video file path')
    parser.add_argument('-w', '--width', type=int, default=1920, help='Video width')
    parser.add_argument('-b', '--height', type=int, default=1080, help='Video height')
    parser.add_argument('-f', '--fps', type=int, default=50, help='Video frame rate')
    parser.add_argument('--no-audio', action='store_true', help='Skip adding audio to output video')
    parser.add_argument('--text1', help='First line of text to display above visualization')
    parser.add_argument('--text2', help='Second line of text to display above visualization')

    args = parser.parse_args()

    if not os.path.exists(args.input_audio):
        print(f"Error: Audio file '{args.input_audio}' not found")
        sys.exit(1)

    # Create visualizer
    visualizer = VideoMusicVisualizer(width=args.width, height=args.height, fps=args.fps, 
                                    text_line1=args.text1, text_line2=args.text2)

    # Load and process audio
    if not visualizer.load_audio(args.input_audio):
        print("Failed to load audio file")
        sys.exit(1)

    # Render video
    audio_path = None if args.no_audio else args.input_audio
    if visualizer.render_video(args.output, audio_path):
        print(f"\nVisualization complete! Output saved to: {args.output}")
    else:
        print("Failed to render video")
        sys.exit(1)

if __name__ == "__main__":
    main()
