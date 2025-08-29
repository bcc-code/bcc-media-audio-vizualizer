import numpy as np
import librosa
import cv2
import sys
import os
from PIL import Image, ImageDraw
import argparse

class VideoMusicVisualizer:
    def __init__(self, width=1920, height=1080, fps=50):
        self.width = width
        self.height = height
        self.fps = fps

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
        self.spectrum_data = []

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
        """Load and analyze audio file"""
        try:
            print("Loading audio file...")
            # Load audio with librosa
            self.audio_data, self.sample_rate = librosa.load(file_path, sr=44100)
            self.duration = len(self.audio_data) / self.sample_rate

            # Pre-compute spectrum data for better performance
            self._precompute_spectrum()

            print(f"Audio loaded: {len(self.audio_data)} samples, {self.sample_rate} Hz, {self.duration:.2f}s")
            return True
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False

    def _precompute_spectrum(self):
        """Pre-compute spectrum data for the entire audio"""
        print("Analyzing audio spectrum...")

        # Window size for FFT (affects frequency resolution vs time resolution)
        window_size = 2048*4
        hop_length = 512*4

        # Compute STFT (Short-Time Fourier Transform)
        stft = librosa.stft(self.audio_data, n_fft=window_size, hop_length=hop_length)
        magnitude = np.abs(stft)

        # Convert to dB scale and normalize
        db_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Select frequency bins (focus on lower frequencies for better visualization)
        freq_bins = np.linspace(0, self.sample_rate // 2, window_size // 2 + 1)
        max_freq = 8000  # Focus on frequencies up to 8kHz
        freq_indices = np.where(freq_bins <= max_freq)[0]

        # Resample to match video frame rate
        total_video_frames = int(self.duration * self.fps)
        spectrum_frames = []

        for frame_idx in range(total_video_frames):
            # Calculate corresponding audio frame
            time_pos = frame_idx / self.fps
            audio_frame_idx = int(time_pos * self.sample_rate / hop_length)

            if audio_frame_idx < db_magnitude.shape[1]:
                frame_spectrum = db_magnitude[freq_indices, audio_frame_idx]
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

            spectrum_frames.append(resampled)

        self.spectrum_data = np.array(spectrum_frames)

        # Normalize spectrum data (0 to 1)
        min_val = np.min(self.spectrum_data)
        max_val = np.max(self.spectrum_data)
        if max_val > min_val:
            self.spectrum_data = (self.spectrum_data - min_val) / (max_val - min_val)

        # Apply smoothing and enhance dynamics
        for i in range(len(self.spectrum_data)):
            # Enhance the spectrum by applying power scaling
            self.spectrum_data[i] = np.power(self.spectrum_data[i], 0.7)

        print(f"Audio analysis complete! Generated {len(self.spectrum_data)} video frames")

    def create_frame(self, spectrum):
        """Create a single frame of the visualization"""
        # Create PIL image for easier drawing
        img = Image.new('RGB', (self.width, self.height), self.bg_color)
        draw = ImageDraw.Draw(img)

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

    def render_video(self, output_path, audio_path=None):
        """Render the complete visualization to video"""
        if len(self.spectrum_data) == 0:
            print("No audio data loaded!")
            return False

        print(f"Rendering video: {output_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {len(self.spectrum_data)}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            return False

        try:
            for frame_idx, spectrum in enumerate(self.spectrum_data):
                # Create frame
                frame_rgb = self.create_frame(spectrum)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Write frame
                video_writer.write(frame_bgr)

                # Progress indicator
                if frame_idx % (len(self.spectrum_data) // 10) == 0:
                    progress = (frame_idx / len(self.spectrum_data)) * 100
                    print(f"Progress: {progress:.1f}%")

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

    args = parser.parse_args()

    if not os.path.exists(args.input_audio):
        print(f"Error: Audio file '{args.input_audio}' not found")
        sys.exit(1)

    # Create visualizer
    visualizer = VideoMusicVisualizer(width=args.width, height=args.height, fps=args.fps)

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
