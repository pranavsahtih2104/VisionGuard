"""
VisionGuard — Real-Time Deep Learning Surveillance System
Entry point: Flask web server + live recognition stream
"""

from flask import Flask, render_template, Response, jsonify, request
from src.recognizer import FaceRecognizer
from src.alert import AlertSystem
from database.db import Database
import cv2
import os

app = Flask(__name__)

db       = Database()
recognizer = FaceRecognizer(db)
alert_sys  = AlertSystem(db)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    stats = db.get_stats()
    logs  = db.get_recent_logs(20)
    return render_template('dashboard.html', stats=stats, logs=logs)

@app.route('/video_feed')
def video_feed():
    """MJPEG stream with live face annotations."""
    return Response(
        recognizer.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/stats')
def api_stats():
    return jsonify(db.get_stats())

@app.route('/api/logs')
def api_logs():
    return jsonify(db.get_recent_logs(50))

@app.route('/api/register', methods=['POST'])
def register_face():
    """Register a new person from an uploaded image."""
    data = request.get_json()
    name   = data.get('name', '').strip()
    image_b64 = data.get('image')
    if not name or not image_b64:
        return jsonify({'success': False, 'error': 'Name and image required'}), 400
    result = recognizer.register_person(name, image_b64)
    return jsonify(result)

@app.route('/api/alerts')
def api_alerts():
    return jsonify(db.get_alerts(20))

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("VisionGuard starting...")
    print("Open http://localhost:5000 in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
