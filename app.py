from flask import Flask, request, jsonify
import os
import uuid
import threading
import time
from video_visualizer import VideoMusicVisualizer

app = Flask(__name__)

# Global job tracking
jobs = {}

class JobStatus:
    def __init__(self, job_id):
        self.job_id = job_id
        self.status = 'pending'
        self.progress = 0
        self.message = 'Job created'
        self.output_file = None
        self.error = None
        self.created_at = time.time()

def generate_visualization_worker(job_id, audio_path, output_path, width, height, fps, include_audio):
    """Background worker to generate visualization"""
    try:
        job = jobs[job_id]
        job.status = 'processing'
        job.message = 'Initializing visualizer'
        
        # Create visualizer
        visualizer = VideoMusicVisualizer(width=width, height=height, fps=fps)
        
        job.message = 'Loading audio file'
        job.progress = 10
        
        # Load audio
        if not visualizer.load_audio(audio_path):
            job.status = 'failed'
            job.error = 'Failed to load audio file'
            return
        
        job.message = 'Rendering visualization'
        job.progress = 30
        
        # Render video
        audio_for_output = audio_path if include_audio else None
        if visualizer.render_video(output_path, audio_for_output):
            job.status = 'completed'
            job.progress = 100
            job.message = 'Visualization completed successfully'
            job.output_file = output_path
        else:
            job.status = 'failed'
            job.error = 'Failed to render video'
            
    except Exception as e:
        job.status = 'failed'
        job.error = str(e)

@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """Create visualization from local audio file"""
    data = request.get_json()
    
    if not data or 'audio_path' not in data:
        return jsonify({'error': 'audio_path is required'}), 400
    
    audio_path = data['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({'error': 'Audio file not found'}), 404
    
    # Extract parameters with defaults
    output_path = data.get('output_path', f'visualization_{uuid.uuid4()}.mp4')
    width = data.get('width', 1920)
    height = data.get('height', 1080)
    fps = data.get('fps', 50)
    include_audio = data.get('include_audio', True)
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create job status
    jobs[job_id] = JobStatus(job_id)
    
    # Start background processing
    thread = threading.Thread(
        target=generate_visualization_worker,
        args=(job_id, audio_path, output_path, width, height, fps, include_audio)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'pending',
        'message': 'Visualization job started',
        'output_path': output_path
    })

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of visualization job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job.status,
        'progress': job.progress,
        'message': job.message,
        'created_at': job.created_at
    }
    
    if job.output_file:
        response['output_file'] = job.output_file
    
    if job.error:
        response['error'] = job.error
    
    return jsonify(response)

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    job_list = []
    for job_id, job in jobs.items():
        job_info = {
            'job_id': job_id,
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
            'created_at': job.created_at
        }
        if job.output_file:
            job_info['output_file'] = job.output_file
        job_list.append(job_info)
    
    return jsonify({'jobs': job_list})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Music Visualizer API',
        'version': '1.0.0'
    })

@app.route('/')
def index():
    """Simple API documentation"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Music Visualizer API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #007acc; }
            pre { background: #eee; padding: 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>Music Visualizer API</h1>
        <p>REST API for generating music visualizations from local audio files.</p>
        
        <div class="endpoint">
            <div class="method">POST /api/visualize</div>
            <p>Create visualization from local audio file</p>
            <p><strong>Body (JSON):</strong></p>
            <pre>{
  "audio_path": "/path/to/audio.mp3",
  "output_path": "/path/to/output.mp4",  // optional
  "width": 1920,      // optional, default: 1920
  "height": 1080,     // optional, default: 1080  
  "fps": 50,          // optional, default: 50
  "include_audio": true  // optional, default: true
}</pre>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /api/status/{job_id}</div>
            <p>Check status of visualization job</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /api/jobs</div>
            <p>List all jobs</p>
        </div>
        
        <div class="endpoint">
            <div class="method">GET /api/health</div>
            <p>Health check</p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)