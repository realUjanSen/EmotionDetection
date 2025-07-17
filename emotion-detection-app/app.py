from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import base64
import numpy as np
import cv2
from utils.db import register_user, login_user, log_emotion, get_user_by_email, get_recent_emotion_logs, get_user_stats, get_recent_sessions_with_details
from utils.face_detector import FaceDetector
from utils.emotion_predictor import EmotionPredictor

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and detectors once at startup (PERFORMANCE OPTIMIZATION)
print("🚀 Loading face detector and emotion model...")
global_face_detector = FaceDetector()
global_emotion_predictor = EmotionPredictor('models/fer2013_big_XCEPTION.54-0.66.hdf5')
print("✅ Models loaded successfully!")

@app.route('/')
def index():
    alert = session.pop('alert_message', None)
    return render_template('index.html', alert=alert)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'logged_in' in session and session['logged_in']:
        print("[DEBUG] User already logged in. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        register_user(name, email, password)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'logged_in' in session and session['logged_in']:
        print("[DEBUG] User already logged in. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if login_user(email, password):
            session['email'] = email
            session['logged_in'] = True
            print(f"[DEBUG] Login successful for user: {email}")
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid email or password.'
            print("[DEBUG] Login failed. Invalid credentials.")
    return render_template('login.html', error=error)

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        user = get_user_by_email(session['email'])
        
        # Get dynamic user statistics
        user_stats = get_user_stats(user['id'])
        
        # Get recent emotion sessions with proper formatting
        recent_sessions = get_recent_sessions_with_details(user['id'], limit=10)
        
        # Provide default empty data for emotion_data (for chart)
        emotion_data = {"labels": [], "values": []}
        
        return render_template(
            'dashboard.html', 
            user=user, 
            user_stats=user_stats,
            recent_sessions=recent_sessions,
            emotion_data=emotion_data, 
            emotion_logs=recent_sessions  # Keep for backwards compatibility
        )
    return redirect(url_for('login'))

@app.route('/track', methods=['GET', 'POST'])
def track():
    # Provide placeholder data for the chart and emotion logs table
    labels = ['10:30', '10:35', '10:40', '10:45', '10:50', '10:55', '11:00']
    data = [3, 7, 2, 5, 8, 4, 6]  # Sample emotion intensity data
    
    # Create some placeholder emotion logs for demo
    emotion_logs = [
        {'date': '2025-06-29 11:05:23', 'emotion': 'Happy'},
        {'date': '2025-06-29 11:04:45', 'emotion': 'Neutral'}, 
        {'date': '2025-06-29 11:03:12', 'emotion': 'Surprise'},
        {'date': '2025-06-29 11:02:38', 'emotion': 'Happy'},
        {'date': '2025-06-29 11:01:55', 'emotion': 'Neutral'},
        {'date': '2025-06-29 11:00:21', 'emotion': 'Sad'},
        {'date': '2025-06-29 10:59:47', 'emotion': 'Happy'},
        {'date': '2025-06-29 10:58:15', 'emotion': 'Fear'},
        {'date': '2025-06-29 10:57:33', 'emotion': 'Neutral'},
        {'date': '2025-06-29 10:56:12', 'emotion': 'Happy'},
    ]
    
    if request.method == 'POST':
        video_capture = cv2.VideoCapture(0)
        detector = FaceDetector()
        predictor = EmotionPredictor('models/fer2013_big_XCEPTION.54-0.66.hdf5')  # Updated model path
        while True:
            ret, frame = video_capture.read()
            faces = detector.detect_faces(frame)
            for face_coords in faces:
                x1, y1, x2, y2 = face_coords
                face = frame[y1:y2, x1:x2]
                emotion, confidence = predictor.predict_emotion(face)
                log_emotion(session['email'], emotion)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
    return render_template('track.html', labels=labels, data=data, emotion_logs=emotion_logs)

@app.route('/trends')
def trends():
    if 'logged_in' not in session or not session['logged_in']:
        session['alert_message'] = 'You cannot access this page. You need to log in or register first.'
        return redirect(url_for('index'))
    # Provide default empty data for emotion_data
    emotion_data = []
    return render_template('trends.html', emotion_data=emotion_data)

@app.route('/logout')
def logout():
    session.pop('email', None)
    session.pop('logged_in', None)
    session['alert_message'] = 'You have been logged out successfully.'
    print("[DEBUG] User logged out. Session cleared.")
    return redirect(url_for('index'))

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame uploaded'}), 400
    file = request.files['frame']
    # Read image as numpy array
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Use global instances (MUCH FASTER)
    faces = global_face_detector.detect_faces(img)
    emotion = 'No face detected'
    
    if faces:
        for (x1, y1, x2, y2) in faces:
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_img = img[y1:y2, x1:x2]
            # Predict emotion if face region is valid
            if face_img.size > 0:
                emotion, _ = global_emotion_predictor.predict_emotion(face_img)
                # Draw label with larger font size
                cv2.putText(img, emotion, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    else:
        emotion = 'No face detected'

    # Encode image to base64 with higher JPEG quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    _, buffer = cv2.imencode('.jpg', img, encode_param)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': img_base64, 'emotion': emotion})

@app.route('/capture_emotion', methods=['POST'])
def capture_emotion():
    if 'email' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    emotion = request.json.get('emotion')
    if not emotion:
        return jsonify({'error': 'No emotion provided'}), 400

    # Get user by email to retrieve user_id
    user = get_user_by_email(session['email'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user_id = user['id']

    log_emotion(user_id, emotion)
    print(f"[DEBUG] Captured emotion logged: {emotion} for user_id: {user_id}")
    return jsonify({'success': True, 'message': 'Emotion logged successfully'})

@app.route('/api/emotion_logs')
def api_emotion_logs():
    """API endpoint to get recent emotion logs for the current user"""
    if 'email' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    # Get user by email to retrieve user_id
    user = get_user_by_email(session['email'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    user_id = user['id']
    
    # Get last 25 emotion logs
    logs = get_recent_emotion_logs(user_id, 25)
    
    # Format data for the frontend (similar to track.html format)
    timestamps = []
    emotions = []
    
    for log in logs:
        # Format timestamp for display
        timestamps.append(log['timestamp'].strftime('%H:%M:%S'))
        emotions.append(log['emotion'].lower())
    
    return jsonify({
        'timestamps': timestamps,
        'emotions': emotions,
        'logs': [{'date': log['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), 'emotion': log['emotion']} for log in logs]
    })

if __name__ == '__main__':
    app.run(debug=True)