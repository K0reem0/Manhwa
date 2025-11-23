# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
import uuid
import math
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename
import io
import zipfile
import shutil

eventlet.monkey_patch()

load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
MAX_SPLIT_HEIGHT = 5000  # أقصى ارتفاع للجزء الواحد

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # رفع الحد لـ 100 ميجا للصور الطويلة

# --- SocketIO Setup ---
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=120, # زيادة وقت الانتظار للملفات الكبيرة
    ping_interval=25
)

# --- Ensure directories exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)

def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60):
    """Fetches predictions from Roboflow."""
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    try:
        response = requests.post(f"{endpoint_url}?api_key={api_key}", data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status()
        return response.json().get("predictions", [])
    except Exception as e:
        print(f"❌ Roboflow Error: {e}")
        raise # نعيد رفع الخطأ ليتم التعامل معه في الدالة الرئيسية

def is_background_white(image, polygon_pts, threshold=180):
    """Checks if the area inside the polygon is mostly white/bright."""
    try:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_pts], 255)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = cv2.mean(gray, mask=mask)[0]
        return mean_val > threshold
    except Exception as e:
        return False

def process_single_chunk(image_chunk, chunk_index, total_chunks, sid):
    """
    دالة مساعدة لمعالجة جزء واحد من الصورة (Detection + Whitening)
    """
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    
    h_img, w_img = image_chunk.shape[:2]
    result_chunk = image_chunk.copy()
    
    # Encode chunk
    retval, buffer_text = cv2.imencode('.jpg', image_chunk)
    if not retval: return image_chunk
    b64_image = base64.b64encode(buffer_text).decode('utf-8')

    try:
        print(f"   Processing chunk {chunk_index + 1}/{total_chunks}...")
        emit_progress(2, f"جاري معالجة الجزء {chunk_index + 1} من {total_chunks}...", 
                      30 + int((chunk_index / total_chunks) * 60), sid)
        
        predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image)
        
        whitened_in_chunk = 0
        for pred in predictions:
            points = pred.get("points", [])
            if len(points) >= 3:
                polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1)
                polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                
                if is_background_white(image_chunk, polygon_np, threshold=200):
                    cv2.fillPoly(result_chunk, [polygon_np], (255, 255, 255))
                    whitened_in_chunk += 1
        
        return result_chunk

    except Exception as e:
        print(f"⚠️ Error processing chunk {chunk_index}: {e}")
        # في حال الفشل في جزء معين، نعيد الجزء الأصلي دون تعديل لتجنب فشل الصورة بالكامل
        return image_chunk

# --- Main Processing Task ---
def process_image_task(image_path, output_filename_base, mode, sid):
    print(f"ℹ️ SID {sid}: Starting task for {os.path.basename(image_path)}")
    start_time = time.time()
    final_output_path = ""
    result_data = {}

    try:
        emit_progress(0, "تحميل الصورة...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path}")
        
        # Ensure 3 channels
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        h_full, w_full = image.shape[:2]
        
        # --- منطق التقسيم (Slicing Logic) ---
        if h_full > MAX_SPLIT_HEIGHT:
            print(f"ℹ️ Image height ({h_full}px) exceeds limit ({MAX_SPLIT_HEIGHT}px). Splitting...")
            emit_progress(1, "الصورة طويلة جداً، جاري تقسيمها...", 10, sid)
            
            processed_chunks = []
            num_chunks = math.ceil(h_full / MAX_SPLIT_HEIGHT)
            
            for i in range(num_chunks):
                y_start = i * MAX_SPLIT_HEIGHT
                y_end = min((i + 1) * MAX_SPLIT_HEIGHT, h_full)
                
                # اقتطاع الجزء
                chunk = image[y_start:y_end, :]
                
                # معالجة الجزء
                processed_chunk = process_single_chunk(chunk, i, num_chunks, sid)
                processed_chunks.append(processed_chunk)
            
            print("ℹ️ Merging chunks back together...")
            emit_progress(4, "دمج الأجزاء...", 90, sid)
            # دمج الأجزاء رأسياً
            final_image = cv2.vconcat(processed_chunks)
            
        else:
            # معالجة عادية للصورة الصغيرة
            final_image = process_single_chunk(image, 0, 1, sid)

        # Save Result
        output_filename = f"{output_filename_base}_cleaned.jpg"
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        emit_progress(5, "حفظ الصورة النهائية...", 95, sid)
        cv2.imwrite(final_output_path, final_image)
        
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Complete {processing_time:.2f}s.")
        emit_progress(6, "تم!", 100, sid)
        
        result_data = {
            'mode': 'clean_white', 
            'imageUrl': f'/results/{output_filename}',
            'original_filename': os.path.basename(image_path)
        }
        socketio.emit('processing_complete', result_data, room=sid)

    except Exception as e:
        print(f"❌❌❌ SID {sid}: FATAL ERROR: {e}")
        traceback.print_exc()
        emit_error(f"خطأ في الخادم: {str(e)}", sid)
    finally:
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except: pass

# --- Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/results/<path:filename>')
def get_result_image(filename):
    return send_from_directory(os.path.abspath(app.config['RESULT_FOLDER']), filename)

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS: return jsonify({'error': 'Invalid file type'}), 400
    
    unique_id = uuid.uuid4()
    input_filename = f"{unique_id}.{ext}"
    output_filename_base = f"{unique_id}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    
    file.save(upload_path)
    return jsonify({'message': 'OK', 'output_filename_base': output_filename_base, 'saved_filename': input_filename}), 200

@app.route('/upload_zip', methods=['POST'])
def handle_zip_upload():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.zip'): return jsonify({'error': 'Not a zip file'}), 400

    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_zip_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(io.BytesIO(file.read())) as z:
            extracted = []
            for info in z.infolist():
                if info.is_dir(): continue
                fname = os.path.basename(info.filename)
                ext = fname.rsplit('.', 1)[1].lower() if '.' in fname else ''
                if ext in ALLOWED_EXTENSIONS:
                    path = os.path.join(temp_dir, secure_filename(fname))
                    with open(path, "wb") as f: f.write(z.read(info.filename))
                    extracted.append({
                        'original_filename': fname,
                        'saved_path': path,
                        'output_base': f"zip_{uuid.uuid4()}"
                    })
        return jsonify({'message': 'OK', 'images_to_process': extracted}), 200
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

# --- Socket Events ---
@socketio.on('start_processing')
def handle_start_processing(data):
    sid = request.sid
    if not data.get('saved_filename'): return
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(data['saved_filename']))
    socketio.start_background_task(process_image_task, path, data['output_filename_base'], 'clean_white', sid)

@socketio.on('start_batch_processing')
def handle_batch(data):
    sid = request.sid
    images = data.get('images_to_process', [])
    socketio.emit('batch_started', {'total_images': len(images)}, room=sid)
    for img in images:
        socketio.start_background_task(process_image_task, img['saved_path'], img['output_base'], 'clean_white', sid)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
