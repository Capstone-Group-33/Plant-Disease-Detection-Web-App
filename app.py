from flask import Flask, render_template_string, Response, jsonify, request
import cv2
import threading
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import queue
import os
from werkzeug.utils import secure_filename

# Create app
app = Flask(__name__)

# Uploads folder
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

# Disease diagnosis and remedy dictionary
disease_info = {
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "diagnosis": "A viral disease spread by whiteflies, causing curling and yellowing of leaves with stunted growth.",
        "remedy": "Remove infected plants, control whiteflies using sticky traps or neem oil, and plant resistant varieties."
    },
    "Tomato_Mosaic_Virus": {
        "diagnosis": "Viral infection leading to mottled, discolored leaves and reduced fruit quality.",
        "remedy": "Remove infected plants, disinfect tools, and wash hands before handling plants (avoid tobacco exposure)."
    },
    "Tomato_Target_Spot": {
        "diagnosis": "Fungal disease causing brown concentric spots on leaves and fruit.",
        "remedy": "Prune lower leaves, improve air circulation, and apply copper-based fungicide."
    },
    "Tomato_Spider_Mites": {
        "diagnosis": "Tiny mites that cause yellow stippling and webbing on leaves.",
        "remedy": "Spray leaves with water, neem oil, or insecticidal soap. Encourage natural predators like ladybugs."
    },
    "Tomato_Septoria_Leaf_Spot": {
        "diagnosis": "Fungal infection causing small circular spots with dark borders on lower leaves.",
        "remedy": "Remove infected leaves, avoid wetting foliage, and apply fungicide like mancozeb or chlorothalonil."
    },
    "Tomato_Leaf_Mold": {
        "diagnosis": "High humidity fungal disease causing yellow spots and mold growth on leaves' underside.",
        "remedy": "Increase ventilation, reduce humidity, and treat with sulfur or copper fungicides."
    },
    "Tomato_Late_Blight": {
        "diagnosis": "Serious fungal disease causing dark, water-soaked lesions on leaves and fruit.",
        "remedy": "Destroy infected plants, avoid overhead watering, and apply fungicides containing chlorothalonil."
    },
    "Tomato_Healthy": {
        "diagnosis": "No signs of disease. Plant appears healthy and vigorous.",
        "remedy": "Continue regular care—ensure balanced nutrients and pest monitoring."
    },
    "Tomato_Early_Blight": {
        "diagnosis": "Fungal disease causing dark, concentric leaf spots that start on lower leaves.",
        "remedy": "Remove affected leaves, rotate crops, and spray with fungicides like mancozeb."
    },
    "Tomato_Bacterial_Spot": {
        "diagnosis": "Bacterial infection causing water-soaked lesions on leaves and fruits.",
        "remedy": "Avoid overhead watering, use copper-based bactericides, and destroy infected debris."
    },
    "Potato_Healthy": {
        "diagnosis": "No visible infection detected. Plant is healthy.",
        "remedy": "Maintain good soil health, avoid overwatering, and monitor for pests."
    },
    "Potato_Late_Blight": {
        "diagnosis": "Serious fungal disease leading to dark lesions and tuber rot.",
        "remedy": "Remove infected plants, avoid wet foliage, and use preventive fungicides regularly."
    },
    "Potato_Early_Blight": {
        "diagnosis": "Dark spots with concentric rings that lead to leaf drop.",
        "remedy": "Remove infected leaves, apply fungicide, and ensure crop rotation."
    },
    "Corn_Healthy": {
        "diagnosis": "No disease detected.",
        "remedy": "Maintain field hygiene, balanced nutrition, and adequate spacing."
    },
    "Corn_Gray_Leaf_Spot": {
        "diagnosis": "Gray or tan rectangular lesions caused by Cercospora fungus.",
        "remedy": "Use resistant hybrids, rotate crops, and apply fungicides at early tasseling."
    },
    "Corn_Common_Rust": {
        "diagnosis": "Small reddish-brown pustules on both sides of leaves.",
        "remedy": "Use rust-resistant hybrids and apply fungicides when infection is severe."
    },
    "Corn_Blight": {
        "diagnosis": "Fungal leaf disease causing elongated gray or tan lesions that reduce yield.",
        "remedy": "Use resistant varieties, rotate crops, and remove infected residues."
    },
    "Rice_Brown_Spot": {
        "diagnosis": "Fungal disease causing small brown spots on leaves and grains.",
        "remedy": "Apply balanced fertilizers, improve drainage, and spray fungicide if needed."
    },
    "Rice_Leaf_Smut": {
        "diagnosis": "Fungal infection forming black, dusty smut balls on leaves.",
        "remedy": "Use disease-free seeds, avoid excessive nitrogen fertilizer, and treat with carbendazim."
    },
    "Rice_Bacterial_Leaf_Blight": {
        "diagnosis": "Bacterial disease causing yellowing and wilting of leaves from tip downward.",
        "remedy": "Use resistant varieties, avoid mechanical injury, and apply copper-based bactericide."
    },
}

# HTML template with updated live detection display
HTML_TEMPLATE = '''<!DOCTYPE html>
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
            max-height: 500px;
            overflow-y: auto;
        }
        
        .detections-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #2d3748;
        }
        
        .live-disease-item {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .live-disease-item:last-child {
            margin-bottom: 0;
        }
        
        .live-disease-name {
            font-size: 1rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .live-confidence-badge {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
            padding: 3px 10px;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .live-section {
            margin-bottom: 10px;
        }
        
        .live-section:last-child {
            margin-bottom: 0;
        }
        
        .live-section-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 5px;
        }
        
        .live-section-content {
            font-size: 0.85rem;
            line-height: 1.5;
            color: #718096;
        }
        
        .no-detection-message {
            text-align: center;
            padding: 30px;
            color: #667eea;
            font-weight: 500;
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
        
        .diagnosis-remedy-container {
            background: white;
            border-radius: 12px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .diagnosis-section, .remedy-section {
            margin-bottom: 20px;
        }
        
        .diagnosis-section:last-child, .remedy-section:last-child {
            margin-bottom: 0;
        }
        
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .section-content {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            line-height: 1.6;
            color: #4a5568;
            border-left: 4px solid #667eea;
        }
        
        .disease-item {
            margin-bottom: 25px;
            padding-bottom: 25px;
            border-bottom: 2px solid #e2e8f0;
        }
        
        .disease-item:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }
        
        .disease-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .confidence-badge {
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
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
            <p>Advanced Plant Disease Detection with Real-time Analysis</p>
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
                    <div id="detections">
                        <div class="no-detection-message">Waiting for detections...</div>
                    </div>
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

        // Poll detections with diagnosis and remedy display
        async function pollDetections() {
            while(true) {
                try {
                    const response = await fetch('/detections');
                    const data = await response.json();
                    
                    const detectionsDiv = document.getElementById("detections");
                    
                    if (data.detections && data.detections.length > 0) {
                        let html = '';
                        data.detections.forEach((det, index) => {
                            html += `
                                <div class="live-disease-item">
                                    <div class="live-disease-name">
                                        🦠 ${det.class}
                                        <span class="live-confidence-badge">${(det.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div class="live-section">
                                        <div class="live-section-title">🔬 Diagnosis:</div>
                                        <div class="live-section-content">${det.diagnosis || 'Information not available'}</div>
                                    </div>
                                    <div class="live-section">
                                        <div class="live-section-title">💊 Remedy:</div>
                                        <div class="live-section-content">${det.remedy || 'Information not available'}</div>
                                    </div>
                                </div>
                            `;
                        });
                        detectionsDiv.innerHTML = html;
                    } else {
                        detectionsDiv.innerHTML = '<div class="no-detection-message">👀 No diseases detected</div>';
                    }
                } catch (error) {
                    document.getElementById("detections").innerHTML = `<div class="no-detection-message" style="color: #ff6b6b;">❌ Connection error: ${error.message}</div>`;
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
                    let diagnosisHTML = '';
                    
                    if (data.detections && data.detections.length > 0) {
                        diagnosisHTML = data.detections.map((det, index) => `
                            <div class="disease-item">
                                <div class="disease-name">
                                    🦠 ${det.class}
                                    <span class="confidence-badge">${(det.confidence * 100).toFixed(1)}% confidence</span>
                                </div>
                                <div class="diagnosis-section">
                                    <div class="section-title">
                                        🔬 Diagnosis
                                    </div>
                                    <div class="section-content">
                                        ${det.Diagnosis || 'Information not available'}
                                    </div>
                                </div>
                                <div class="remedy-section">
                                    <div class="section-title">
                                        💊 Recommended Treatment
                                    </div>
                                    <div class="section-content">
                                        ${det.Remedy || 'Information not available'}
                                    </div>
                                </div>
                            </div>
                        `).join('');
                    }
                    
                    uploadResult.innerHTML = `
                        <div class="results">
                            <div class="detection-summary">
                                <div class="detection-count">${data.detections.length}</div>
                                <div>${data.detections.length === 1 ? 'Disease' : 'Diseases'} Detected</div>
                            </div>
                            
                            ${data.detections.length > 0 ? `
                                <div class="diagnosis-remedy-container">
                                    ${diagnosisHTML}
                                </div>
                            ` : '<p style="text-align: center; color: #667eea; font-size: 1.1rem; margin: 20px 0;">No diseases detected - Plant appears healthy! 🌱</p>'}
                            
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

        console.log("🎯 Plant Disease Detection App loaded successfully!");
    </script>
</body>
</html>''' 

# Model loading
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

# Camera thread
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

                if model_obj['loaded'] and model_obj['model']:
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
                                    
                                    # Include diagnosis & remedy
                                    info = disease_info.get(name.replace(" ", "_"), {
                                        'diagnosis': 'Info not available',
                                        'remedy': 'Info not available'
                                    })
                                    
                                    local_detections.append({
                                        'class': name,
                                        'confidence': conf_val,
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'diagnosis': info['diagnosis'],
                                        'remedy': info['remedy']
                                    })
                                    
                                    color = tuple(int(x) for x in np.random.RandomState(cls).randint(0, 255, 3))
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(annotated, f"{name}: {conf_val:.2f}", 
                                              (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Detection error: {e}")

                global detections
                detections = local_detections

                ret2, buf = cv2.imencode('.jpg', annotated)
                if ret2:
                    if frame_queue.full():
                        try: frame_queue.get_nowait()
                        except queue.Empty: pass
                    frame_queue.put(buf.tobytes())
                
                time.sleep(0.03)
        finally:
            if self.cap: self.cap.release()
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
        while not frame_queue.empty():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
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
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if not model_obj['loaded'] or not model_obj['model']:
        return jsonify({'success': False, 'message': 'YOLO model not loaded yet. Please wait.'})

    try:
        results = model_obj['model'](filepath, conf=0.5, verbose=False)
        annotated = results[0].plot()
        output_filename = f"annotated_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated)

        local_detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0])
                cls = int(box.cls[0])
                name = model_obj['model'].names.get(cls, str(cls))
                
                info = disease_info.get(name.replace(" ", "_"), {
                    'diagnosis': 'Info not available',
                    'remedy': 'Info not available'
                })
                
                local_detections.append({
                    'class': name,
                    'confidence': conf_val,
                    'Diagnosis': info['diagnosis'],
                    'Remedy': info['remedy']
                })

        return jsonify({
            'success': True,
            'detections': local_detections,
            'input_image': f"/{UPLOAD_FOLDER}/{filename}",
            'output_image': f"/{UPLOAD_FOLDER}/{output_filename}"
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Detection failed: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    print("="*50)
    print("Plant Disease Detection App Starting...")
    print("URL: http://127.0.0.1:8000")
    print("Make sure 'best.pt' is in the same directory!")
    print("="*50)
    app.run(host="0.0.0.0", port=8000, debug=True)