from flask import Flask, render_template_string, Response, jsonify, request, redirect, url_for
import cv2
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import queue
import os, json
from werkzeug.utils import secure_filename

# Create app with default settings
app = Flask(__name__)

# Create uploads directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables
frame_queue = queue.Queue(maxsize=2)
detections = []
model_obj = {'model': None, 'loading': False, 'loaded': False}
stop_event = threading.Event()
camera_thread = None
camera_lock = threading.Lock()

# HTML template as string (no separate file needed)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎯 Leaf Scan App </title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1200px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 60px rgba(0,0,0,0.15);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 25px;
        }
        
        .card-icon {
            font-size: 2rem;
            margin-right: 15px;
        }
        
        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2d3748;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
        }
        
        .status {
            padding: 12px 20px;
            border-radius: 10px;
            font-weight: 600;
            margin: 15px 0;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .status.success {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
        }
        
        .status.error {
            background: linear-gradient(135deg, #ff6b6b 0%, #fa5252 100%);
            color: white;
        }
        
        .status.warning {
            background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
            color: #333;
        }
        
        .video-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin: 20px 0;
        }
        
        #video {
            width: 100%;
            height: auto;
            display: block;
        }
        
        .detections-container {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }
        
        .detections-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2d3748;
        }
        
        #detections {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            font-size: 0.85rem;
            max-height: 250px;
            overflow-y: auto;
            line-height: 1.4;
            font-family: 'Courier New', monospace;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            transition: all 0.3s ease;
            margin: 20px 0;
        }
        
        .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        }
        
        .upload-icon {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 20px;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 1rem;
            margin: 15px 0;
            background: white;
            transition: border-color 0.3s ease;
        }
        
        input[type="file"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .results {
            margin-top: 30px;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }
        
        @media (max-width: 768px) {
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .result-item {
            text-align: center;
        }
        
        .result-label {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2d3748;
        }
        
        .result-image {
            width: 100%;
            max-width: 400px;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }
        
        .result-image:hover {
            transform: scale(1.02);
        }
        
        .detection-summary {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }
        
        .detection-count {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .detection-details {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
        }
        
        .loading .spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #e2e8f0;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Leaf Sense App</h1>
            <p>Advanced Object Detection with Real-time Analysis</p>
        </div>
        
        <div class="grid">
            <!-- Live Camera Section -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📹</div>
                    <div class="card-title">Live Camera Detection</div>
                </div>
                
                <div class="controls">
                    <button class="btn-primary" onclick="startCamera()">
                        ▶️ Start Camera
                    </button>
                    <button class="btn-danger" onclick="stopCamera()">
                        ⏹️ Stop Camera
                    </button>
                </div>
                
                <div id="status" class="status" style="display: none;"></div>
                
                <div class="video-container">
                    <img id="video" src="/video_feed" alt="Camera feed will appear here">
                </div>
                
                <div class="detections-container">
                    <div class="detections-header">🔍 Live Detection Results</div>
                    <pre id="detections">Waiting for detections...</pre>
                </div>
            </div>
            
            <!-- Upload Section -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon">📤</div>
                    <div class="card-title">Image Upload & Analysis</div>
                </div>
                
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="upload-area">
                        <div class="upload-icon">🖼️</div>
                        <p style="margin-bottom: 20px; color: #667eea; font-weight: 600;">
                            Drop your image here or click to browse
                        </p>
                        <input type="file" name="file" accept="image/*" required>
                        <br>
                        <button type="submit" class="btn-primary" style="margin-top: 15px;">
                            🔍 Analyze Image
                        </button>
                    </div>
                </form>
                
                <div id="uploadResult"></div>
            </div>
        </div>
    </div>

    <script>
        // Camera functions
        async function startCamera() {
            const statusEl = document.getElementById('status');
            statusEl.style.display = 'block';
            
            try {
                const response = await fetch('/start_camera');
                const data = await response.json();
                statusEl.innerText = '✅ ' + data.message;
                statusEl.className = 'status success';
            } catch (error) {
                statusEl.innerText = '❌ Error: ' + error.message;
                statusEl.className = 'status error';
            }
        }

        async function stopCamera() {
            const statusEl = document.getElementById('status');
            statusEl.style.display = 'block';
            
            try {
                const response = await fetch('/stop_camera');
                const data = await response.json();
                statusEl.innerText = '⏹️ ' + data.message;
                statusEl.className = 'status warning';
            } catch (error) {
                statusEl.innerText = '❌ Error: ' + error.message;
                statusEl.className = 'status error';
            }
        }

        // Poll detections with better formatting
        async function pollDetections() {
            while(true) {
                try {
                    const response = await fetch('/detections');
                    const data = await response.json();
                    
                    let formattedText = `🔍 Detection Status: ${data.count} objects found\\n`;
                    formattedText += `⏰ Last Update: ${new Date(data.timestamp).toLocaleTimeString()}\\n\\n`;
                    
                    if (data.detections && data.detections.length > 0) {
                        formattedText += '📋 Detected Objects:\\n';
                        data.detections.forEach((det, index) => {
                            formattedText += `  ${index + 1}. ${det.class} (${(det.confidence * 100).toFixed(1)}%)\\n`;
                        });
                    } else {
                        formattedText += '👀 No objects detected';
                    }
                    
                    document.getElementById("detections").innerText = formattedText;
                } catch (error) {
                    document.getElementById("detections").innerText = "❌ Connection error: " + error.message;
                }
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
        }
        pollDetections();

        // Enhanced upload form
        document.getElementById("uploadForm").addEventListener("submit", async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const uploadResult = document.getElementById("uploadResult");
            
            // Show loading spinner
            uploadResult.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p><strong>🔄 Analyzing your image...</strong></p>
                    <p>This may take a few seconds</p>
                </div>
            `;
            
            try {
                const response = await fetch("/upload", {
                    method: "POST",
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    let detectionsList = '';
                    if (data.detections && data.detections.length > 0) {
                        detectionsList = data.detections.map((det, index) => 
                            `${index + 1}. <strong>${det.class}</strong> - ${(det.confidence * 100).toFixed(1)}% confidence`
                        ).join('<br>');
                    }
                    
                    uploadResult.innerHTML = `
                        <div class="results">
                            <div class="detection-summary">
                                <div class="detection-count">${data.detections.length}</div>
                                <div>Objects Detected</div>
                            </div>
                            
                            ${data.detections.length > 0 ? `
                                <div style="background: white; padding: 20px; border-radius: 12px; margin: 20px 0;">
                                    <h4 style="margin-bottom: 15px; color: #2d3748;">🎯 Detection Results:</h4>
                                    <div style="line-height: 1.8;">${detectionsList}</div>
                                </div>
                            ` : '<p style="text-align: center; color: #667eea; font-size: 1.1rem;">No objects detected in this image</p>'}
                            
                            <div class="results-grid">
                                <div class="result-item">
                                    <div class="result-label">📷 Original Image</div>
                                    <img src="${data.input_image}" class="result-image" alt="Original image">
                                </div>
                                <div class="result-item">
                                    <div class="result-label">🎯 Detection Results</div>
                                    <img src="${data.output_image}" class="result-image" alt="Image with detections">
                                </div>
                            </div>
                            
                            <div class="detection-details">
                                <strong>📊 Technical Details:</strong><br><br>
                                ${JSON.stringify(data.detections, null, 2)}
                            </div>
                        </div>
                    `;
                } else {
                    uploadResult.innerHTML = `
                        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #fa5252 100%); color: white; padding: 30px; border-radius: 15px; text-align: center;">
                            <div style="font-size: 2rem; margin-bottom: 15px;">❌</div>
                            <h3>Analysis Failed</h3>
                            <p>${data.message}</p>
                        </div>
                    `;
                }
            } catch (error) {
                uploadResult.innerHTML = `
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #fa5252 100%); color: white; padding: 30px; border-radius: 15px; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 15px;">🚫</div>
                        <h3>Connection Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        });

        // File input enhancement
        const fileInput = document.querySelector('input[type="file"]');
        fileInput.addEventListener('change', function(e) {
            const uploadArea = document.querySelector('.upload-area');
            if (e.target.files.length > 0) {
                uploadArea.style.borderColor = '#51cf66';
                uploadArea.style.background = 'linear-gradient(135deg, rgba(81, 207, 102, 0.1) 0%, rgba(64, 192, 87, 0.1) 100%)';
            }
        });

        console.log("🎯 YOLO Detection Studio loaded successfully!");
    </script>
</body>
</html>
'''

# Model loading function
def load_model_async(model_path='best.pt'):
    if model_obj['loaded'] or model_obj['loading']:
        return
    model_obj['loading'] = True
    def _loader():
        try:
            print("Loading YOLO model...")
            model = YOLO(model_path)
            model_obj['model'] = model
            model_obj['loaded'] = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Model load error: {e}")
        finally:
            model_obj['loading'] = False
    threading.Thread(target=_loader, daemon=True).start()

# Camera thread class
class CameraThread(threading.Thread):
    def __init__(self, camera_id=0, conf=0.5):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.conf = conf
        self.cap = None
        self.running = False

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.2)
            
            if not self.cap.isOpened():
                print("Camera failed to open")
                return
                
            self.running = True
            print("Camera thread started")
            
            while self.running and not stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                annotated = frame.copy()
                local_detections = []

                # Run detection if model is loaded
                if model_obj['loaded'] and model_obj['model'] is not None:
                    try:
                        results = model_obj['model'](frame, conf=self.conf, verbose=False)
                        for res in results:
                            boxes = res.boxes
                            if boxes is not None:
                                for box in boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    conf_val = float(box.conf[0])
                                    cls = int(box.cls[0])
                                    name = model_obj['model'].names.get(cls, str(cls))
                                    
                                    local_detections.append({
                                        'class': name,
                                        'confidence': conf_val,
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                    })
                                    
                                    # Draw bounding box
                                    color = tuple(int(x) for x in np.random.RandomState(cls).randint(0, 255, 3))
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(annotated, f"{name}: {conf_val:.2f}", 
                                              (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Detection error: {e}")

                # Update global detections
                global detections
                detections = local_detections

                # Encode frame for streaming
                ret2, buf = cv2.imencode('.jpg', annotated)
                if ret2:
                    if frame_queue.full():
                        try: 
                            frame_queue.get_nowait()
                        except queue.Empty: 
                            pass
                    frame_queue.put(buf.tobytes())
                
                time.sleep(0.03)
        finally:
            if self.cap:
                self.cap.release()
            self.running = False
            print("Camera thread stopped")

    def stop(self):
        self.running = False

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/start_camera')
def start_camera():
    global camera_thread
    with camera_lock:
        load_model_async('best.pt')
        if camera_thread is None or not camera_thread.running:
            stop_event.clear()
            camera_thread = CameraThread(camera_id=0)
            camera_thread.start()
            return jsonify({'success': True, 'message': 'Camera started successfully'})
        else:
            return jsonify({'success': True, 'message': 'Camera is already running'})

@app.route('/stop_camera')
def stop_camera():
    global camera_thread
    with camera_lock:
        if camera_thread:
            camera_thread.stop()
            stop_event.set()
            camera_thread = None
        # Clear frame queue
        while not frame_queue.empty():
            try: 
                frame_queue.get_nowait()
            except queue.Empty: 
                pass
        return jsonify({'success': True, 'message': 'Camera stopped'})

def generate_mjpeg():
    while True:
        try:
            frame = frame_queue.get(timeout=5)
        except queue.Empty:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    return jsonify({
        'detections': detections, 
        'count': len(detections), 
        'timestamp': datetime.now().isoformat()
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check if model is loaded
    if not model_obj['loaded'] or model_obj['model'] is None:
        return jsonify({'success': False, 'message': 'YOLO model not loaded yet. Please wait.'})

    try:
        # Run detection
        results = model_obj['model'](filepath, conf=0.5, verbose=False)
        
        # Create annotated image
        annotated = results[0].plot()
        output_filename = f"annotated_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated)

        # Extract detection data
        local_detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0])
                cls = int(box.cls[0])
                name = model_obj['model'].names.get(cls, str(cls))
                local_detections.append({
                    'class': name,
                    'confidence': conf_val,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

        return jsonify({
            'success': True,
            'detections': local_detections,
            'input_image': f"/{UPLOAD_FOLDER}/{filename}",
            'output_image': f"/{UPLOAD_FOLDER}/{output_filename}"
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Detection failed: {str(e)}'})

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return app.send_static_file(f'../uploads/{filename}')

if __name__ == "__main__":
    import os

    print("=" * 50)
    print("YOLO Flask App Starting...")
    port = int(os.environ.get("PORT", 5000))  # ✅ use Render’s assigned port
    print(f"URL: http://0.0.0.0:{port}")
    print("Make sure 'best.pt' is in the same directory!")
    print("=" * 50)

    # ✅ Must bind to 0.0.0.0 for Render
    app.run(debug=False, host="0.0.0.0", port=port)
