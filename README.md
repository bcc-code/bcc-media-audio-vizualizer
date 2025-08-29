# Music Visualizer Video Generator

A Python-based music visualizer that creates stunning bar-style visualizations and renders them to video files. Inspired by modern audio visualization aesthetics with beautiful color gradients.

## Features

- **Bar-style visualization** with frequency-responsive heights
- **Beautiful color gradients** from magenta through purple, blue, cyan, to red/orange
- **High-quality video output** with customizable resolution and frame rate
- **Audio synchronization** with automatic audio track addition
- **Optimized rendering** with pre-computed spectrum analysis

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required for audio integration):
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/

## Usage

### Basic Usage
```bash
python video_visualizer.py input_audio.mp3
```

### Advanced Options
```bash
python video_visualizer.py input_audio.mp3 -o my_visualization.mp4 -w 1920 -b 1080 -f 60
```

### Parameters
- `input_audio`: Path to your audio file (MP3, WAV, FLAC, OGG supported)
- `-o, --output`: Output video filename (default: visualization.mp4)
- `-w, --width`: Video width in pixels (default: 1920)
- `-b, --height`: Video height in pixels (default: 1080)
- `-f, --fps`: Video frame rate (default: 50)
- `--no-audio`: Generate video without audio track

## Examples

Create a 4K visualization at 60fps:
```bash
python video_visualizer.py song.mp3 -o song_4k.mp4 -w 3840 -b 2160 -f 60
```

Generate silent video (video only):
```bash
python video_visualizer.py song.mp3 -o silent_viz.mp4 --no-audio
```

## How It Works

1. **Audio Analysis**: Uses librosa to perform Short-Time Fourier Transform (STFT) on the input audio
2. **Frequency Mapping**: Maps frequency data to visual bars with emphasis on lower frequencies (up to 8kHz)
3. **Color Generation**: Creates smooth color gradients across 120 bars
4. **Frame Generation**: Renders each video frame with PIL for precise drawing
5. **Video Encoding**: Uses OpenCV to encode frames and FFmpeg to add audio

## Technical Details

- **Bars**: 30 frequency bars across the width
- **Frequency Range**: 0-8kHz (optimized for music visualization)
- **Color Palette**: White bars on black background with optional gradient support
- **Audio Processing**: 44.1kHz sample rate with 8192-point FFT
- **Video Codec**: MP4V with AAC audio

## Troubleshooting

### Common Issues

**"Error: Could not open video writer"**
- Ensure you have write permissions in the output directory
- Try a different output filename

**"Warning: Could not add audio"**
- Install FFmpeg: the visualizer needs it to combine video and audio
- Check that FFmpeg is in your system PATH

**Slow rendering**
- Reduce resolution with `-w` and `-b` parameters
- Lower frame rate with `-f` parameter
- Use shorter audio files for testing

### Performance Tips

- **Resolution vs Speed**: 1920x1080 @ 50fps renders much faster than 4K @ 60fps
- **Audio Length**: 3-minute songs typically take 2-5 minutes to render
- **Memory Usage**: Longer songs require more RAM for spectrum analysis

## License

This project is open source. Feel free to modify and distribute.