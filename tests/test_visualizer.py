import pytest
import os
import tempfile
import shutil
import subprocess
import requests
import time
import threading
from video_visualizer import VideoMusicVisualizer
from app import app, jobs

class TestAudioGeneration:
    """Test audio file generation with ffmpeg"""

    @classmethod
    def setup_class(cls):
        """Generate test audio files with ffmpeg"""
        cls.test_dir = tempfile.mkdtemp()

        # Generate mono test file (440Hz sine wave, 3 seconds)
        cls.mono_file = os.path.join(cls.test_dir, 'test_mono.wav')
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
            '-ac', '1', '-ar', '44100', '-y', cls.mono_file
        ], check=True, capture_output=True)

        # Generate stereo test file (440Hz left, 880Hz right, 3 seconds)
        cls.stereo_file = os.path.join(cls.test_dir, 'test_stereo.wav')
        subprocess.run([
            'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',
            '-f', 'lavfi', '-i', 'sine=frequency=880:duration=3',
            '-filter_complex', '[0:a][1:a]amerge=inputs=2[a]',
            '-map', '[a]', '-ac', '2', '-ar', '44100', '-y', cls.stereo_file
        ], check=True, capture_output=True)

        # Generate 6-channel test file (different frequencies per channel)
        cls.multichannel_file = os.path.join(cls.test_dir, 'test_6ch.wav')
        subprocess.run([
            'ffmpeg',
            '-f', 'lavfi', '-i', 'sine=frequency=220:duration=3',  # Channel 1
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=3',  # Channel 2
            '-f', 'lavfi', '-i', 'sine=frequency=660:duration=3',  # Channel 3
            '-f', 'lavfi', '-i', 'sine=frequency=880:duration=3',  # Channel 4
            '-f', 'lavfi', '-i', 'sine=frequency=1100:duration=3', # Channel 5
            '-f', 'lavfi', '-i', 'sine=frequency=1320:duration=3', # Channel 6
            '-filter_complex', '[0:a][1:a][2:a][3:a][4:a][5:a]amerge=inputs=6[a]',
            '-map', '[a]', '-ac', '6', '-ar', '44100', '-y', cls.multichannel_file
        ], check=True, capture_output=True)

    @classmethod
    def teardown_class(cls):
        """Clean up test files"""
        shutil.rmtree(cls.test_dir)

class TestVideoMusicVisualizer(TestAudioGeneration):
    """Test the core VideoMusicVisualizer class"""

    def test_mono_audio_processing(self):
        """Test processing mono audio file"""
        visualizer = VideoMusicVisualizer(width=640, height=480, fps=50)
        assert visualizer.load_audio(self.mono_file)
        assert visualizer.audio_data is not None
        assert visualizer.duration > 0
        assert visualizer.sample_rate == 44100

    def test_stereo_audio_processing(self):
        """Test processing stereo audio file"""
        visualizer = VideoMusicVisualizer(width=640, height=480, fps=50)
        assert visualizer.load_audio(self.stereo_file)
        assert visualizer.audio_data is not None
        assert visualizer.duration > 0

    def test_multichannel_audio_processing(self):
        """Test processing 6-channel audio file"""
        visualizer = VideoMusicVisualizer(width=640, height=480, fps=50)
        assert visualizer.load_audio(self.multichannel_file)
        assert visualizer.audio_data is not None
        assert visualizer.duration > 0

    def test_video_rendering_mono(self):
        """Test complete video rendering with mono audio"""
        output_path = os.path.join('test_outputs', 'test_mono_output.mp4')
        os.makedirs('test_outputs', exist_ok=True)
        
        visualizer = VideoMusicVisualizer(width=320, height=240, fps=50)
        assert visualizer.load_audio(self.mono_file)
        assert visualizer.render_video(output_path, self.mono_file)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        print(f"Mono test output saved to: {output_path}")

    def test_video_rendering_no_audio(self):
        """Test video rendering without audio track"""
        output_path = os.path.join('test_outputs', 'test_silent_output.mp4')
        os.makedirs('test_outputs', exist_ok=True)
        
        visualizer = VideoMusicVisualizer(width=320, height=240, fps=50)
        assert visualizer.load_audio(self.stereo_file)
        assert visualizer.render_video(output_path, None)  # No audio
        assert os.path.exists(output_path)
        print(f"Silent test output saved to: {output_path}")

    def test_invalid_audio_file(self):
        """Test handling of invalid audio file"""
        visualizer = VideoMusicVisualizer()
        assert not visualizer.load_audio('nonexistent_file.mp3')
    
    def test_text_overlay_rendering(self):
        """Test video rendering with text overlay"""
        output_path = os.path.join('test_outputs', 'test_text_overlay.mp4')
        os.makedirs('test_outputs', exist_ok=True)
        
        visualizer = VideoMusicVisualizer(
            width=640, height=480, fps=50,
            text_line1="Test Song Title", 
            text_line2="Test Artist Name"
        )
        assert visualizer.load_audio(self.mono_file)
        assert visualizer.render_video(output_path, self.mono_file)
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0
        print(f"Text overlay test output saved to: {output_path}")
    
    def test_single_text_line(self):
        """Test rendering with only one text line"""
        output_path = os.path.join('test_outputs', 'test_single_text.mp4')
        os.makedirs('test_outputs', exist_ok=True)
        
        visualizer = VideoMusicVisualizer(
            width=640, height=480, fps=50,
            text_line1="Only One Line"
        )
        assert visualizer.load_audio(self.stereo_file)
        assert visualizer.render_video(output_path, None)  # No audio
        assert os.path.exists(output_path)
        print(f"Single text test output saved to: {output_path}")

class TestFlaskAPI(TestAudioGeneration):
    """Test the Flask REST API"""

    @classmethod
    def setup_class(cls):
        """Setup Flask test client and generate audio files"""
        super().setup_class()
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.temp_output_dir = 'test_outputs'
        os.makedirs(cls.temp_output_dir, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        """Clean up"""
        super().teardown_class()
        # Keep test_outputs directory for inspection

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'service' in data

    def test_visualize_endpoint_mono(self):
        """Test visualization creation with mono audio"""
        output_path = os.path.join(self.temp_output_dir, 'api_test_mono.mp4')

        response = self.client.post('/api/visualize', json={
            'audio_path': self.mono_file,
            'output_path': output_path,
            'width': 320,
            'height': 240,
            'fps': 50,
            'include_audio': True
        })

        assert response.status_code == 200
        data = response.get_json()
        assert 'job_id' in data
        assert data['status'] == 'pending'

        job_id = data['job_id']

        # Wait for job completion (with timeout)
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f'/api/status/{job_id}')
            status_data = status_response.get_json()

            if status_data['status'] == 'completed':
                assert 'output_file' in status_data
                break
            elif status_data['status'] == 'failed':
                pytest.fail(f"Job failed: {status_data.get('error', 'Unknown error')}")

            time.sleep(1)
        else:
            pytest.fail("Job did not complete within timeout")

    def test_visualize_endpoint_stereo(self):
        """Test visualization creation with stereo audio"""
        output_path = os.path.join(self.temp_output_dir, 'api_test_stereo.mp4')

        response = self.client.post('/api/visualize', json={
            'audio_path': self.stereo_file,
            'output_path': output_path,
            'width': 320,
            'height': 240,
            'fps': 50,
            'include_audio': False  # No audio for faster processing
        })

        assert response.status_code == 200
        data = response.get_json()
        job_id = data['job_id']

        # Wait for completion
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f'/api/status/{job_id}')
            status_data = status_response.get_json()

            if status_data['status'] in ['completed', 'failed']:
                break
            time.sleep(1)

        assert status_data['status'] == 'completed'

    def test_visualize_endpoint_multichannel(self):
        """Test visualization creation with 6-channel audio"""
        output_path = os.path.join(self.temp_output_dir, 'api_test_6ch.mp4')

        response = self.client.post('/api/visualize', json={
            'audio_path': self.multichannel_file,
            'output_path': output_path,
            'width': 320,
            'height': 240,
            'fps': 50,
            'include_audio': False
        })

        assert response.status_code == 200
        data = response.get_json()
        job_id = data['job_id']

        # Wait for completion
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f'/api/status/{job_id}')
            status_data = status_response.get_json()

            if status_data['status'] in ['completed', 'failed']:
                break
            time.sleep(1)

        assert status_data['status'] == 'completed'

    def test_jobs_endpoint(self):
        """Test jobs listing endpoint"""
        response = self.client.get('/api/jobs')
        assert response.status_code == 200
        data = response.get_json()
        assert 'jobs' in data
        assert isinstance(data['jobs'], list)

    def test_invalid_audio_path(self):
        """Test API with invalid audio path"""
        response = self.client.post('/api/visualize', json={
            'audio_path': '/nonexistent/path.mp3'
        })
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data

    def test_missing_audio_path(self):
        """Test API with missing audio_path parameter"""
        response = self.client.post('/api/visualize', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_invalid_job_id(self):
        """Test status endpoint with invalid job ID"""
        response = self.client.get('/api/status/invalid-job-id')
        assert response.status_code == 404
    
    def test_visualize_with_text_overlay(self):
        """Test API visualization with text overlay"""
        output_path = os.path.join(self.temp_output_dir, 'api_test_text.mp4')
        
        response = self.client.post('/api/visualize', json={
            'audio_path': self.mono_file,
            'output_path': output_path,
            'width': 320,
            'height': 240,
            'fps': 50,
            'include_audio': False,
            'text_line1': 'API Test Song',
            'text_line2': 'Generated by API'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        job_id = data['job_id']
        
        # Wait for completion
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f'/api/status/{job_id}')
            status_data = status_response.get_json()
            
            if status_data['status'] in ['completed', 'failed']:
                break
            time.sleep(1)
        
        assert status_data['status'] == 'completed'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
