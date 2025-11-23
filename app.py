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
MAX_SPLIT_HEIGHT = 15000  # الارتفاع الأقصى قبل التقسيم
SPLIT_OVERLAP = 50       # التداخل بين الأجزاء

# --- نقاط نهاية Roboflow ---
ROBOFLOW_BUBBLE_URL = 'https://serverless.roboflow.com/manga-speech-bubble-detection-1rbgq/15'
ROBOFLOW_TEXT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'

# --- Flask App Setup (كما هو) ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

# --- SocketIO Setup (كما هو) ---
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=120,
    ping_interval=25
)

# --- Ensure directories exist (كما هو) ---
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

def get_roboflow_predictions(endpoint_url, api_key, image_b64, confidence=None, timeout=60):
    """Fetches predictions from Roboflow using the specified endpoint."""
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    
    url = f"{endpoint_url}?api_key={api_key}"
    if confidence is not None:
        url += f"&confidence={confidence}" 
        
    try:
        response = requests.post(url, data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status()
        return response.json().get("predictions", [])
    except Exception as e:
        print(f"❌ Roboflow Error: {e}")
        raise ValueError(f"فشل الاتصال بنموذج Roboflow. تأكد من صحة المفتاح والرابط. الخطأ: {str(e)}")

def get_bounding_box(polygon_np):
    """Calculates min/max x and y from a polygon."""
    x_coords = polygon_np[:, 0]
    y_coords = polygon_np[:, 1]
    return np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)

def expand_polygon_bbox(polygon_pts, image_shape, expansion_factor=0.25):
    """
    *** دالة جديدة: توسيع المربع المحيط حول المضلع بنسبة مئوية (25%) ***
    """
    h_img, w_img = image_shape[:2]
    
    # 1. حساب المربع المحيط الأصلي
    x_min, y_min, x_max, y_max = get_bounding_box(polygon_pts)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # 2. حساب قيمة التوسيع
    # نأخذ أكبر قيمة للبعد (عرض أو ارتفاع) لحساب الهامش لتوسيع موحد
    margin_x = int(width * expansion_factor)
    margin_y = int(height * expansion_factor)
    
    # 3. تطبيق التوسيع مع التأكد من البقاء داخل حدود الصورة
    x_min_exp = max(0, x_min - margin_x)
    y_min_exp = max(0, y_min - margin_y)
    x_max_exp = min(w_img, x_max + margin_x)
    y_max_exp = min(h_img, y_max + margin_y)
    
    # 4. بناء مضلع مستطيل جديد يغطي المنطقة الموسعة
    expanded_bbox_polygon = np.array([
        [x_min_exp, y_min_exp],
        [x_max_exp, y_min_exp],
        [x_max_exp, y_max_exp],
        [x_min_exp, y_max_exp]
    ], dtype=np.int32)
    
    return expanded_bbox_polygon


def is_background_white(image, text_polygon_pts, threshold=190):
    """
    Checks if the area *around* the text polygon is mostly white/bright.
    *** التعديل: يتم الآن فحص المنطقة الموسعة حول النص بنسبة 25% ***
    """
    try:
        # 1. توسيع منطقة فحص النص بنسبة 25%
        expanded_polygon = expand_polygon_bbox(text_polygon_pts, image.shape, expansion_factor=0.25)
        
        # 2. بناء الـ Mask للمنطقة الموسعة
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [expanded_polygon], 255)
        
        # 3. حساب متوسط السطوع داخل الـ Mask
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = cv2.mean(gray, mask=mask)[0]
        
        return mean_val > threshold
    except Exception as e:
        print(f"Error in is_background_white check: {e}")
        return False # الافتراض هو الفشل إذا حدث خطأ

def process_single_chunk(image_chunk, y_offset, chunk_index, total_chunks, sid):
    """
    دالة مساعدة لمعالجة جزء واحد من الصورة (Detection + Whitening)
    """
    result_chunk = image_chunk.copy()
    h_img, w_img = image_chunk.shape[:2]
    
    # 1. الكشف عن الفقاعات باستخدام النموذج الأول (ROBOFLOW_BUBBLE_URL)
    emit_progress(2, f"الكشف عن الفقاعات في الجزء {chunk_index + 1}...", 
                  30 + int((chunk_index / total_chunks) * 20), sid)
    
    retval, buffer_text = cv2.imencode('.png', image_chunk)
    if not retval: return image_chunk
    b64_image = base64.b64encode(buffer_text).decode('utf-8')

    try:
        # استدعاء نموذج الفقاعات بعتبة ثقة 1% (confidence=1)
        bubble_predictions = get_roboflow_predictions(ROBOFLOW_BUBBLE_URL, ROBOFLOW_API_KEY, b64_image, confidence=1)
        
        whitened_count = 0
        
        # 2. المرور على كل فقاعة تم اكتشافها
        for i, bubble_pred in enumerate(bubble_predictions):
            bubble_points = bubble_pred.get("points", [])
            if len(bubble_points) < 3: continue
            
            bubble_polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in bubble_points], dtype=np.int32)
            x_min, y_min, x_max, y_max = get_bounding_box(bubble_polygon_np)

            # 3. اقتصاص الفقاعة لإرسالها لنموذج النص مع هامش موسع (20 بكسل)
            EXPANSION_MARGIN = 20
            
            x_min_exp = max(0, x_min - EXPANSION_MARGIN)
            y_min_exp = max(0, y_min - EXPANSION_MARGIN)
            x_max_exp = min(w_img, x_max + EXPANSION_MARGIN)
            y_max_exp = min(h_img, y_max + EXPANSION_MARGIN)

            cropped_bubble = image_chunk[y_min_exp:y_max_exp, x_min_exp:x_max_exp]
            h_crop, w_crop = cropped_bubble.shape[:2]

            if h_crop == 0 or w_crop == 0: continue
            
            retval_crop, buffer_crop = cv2.imencode('.png', cropped_bubble)
            if not retval_crop: continue
            b64_crop = base64.b64encode(buffer_crop).decode('utf-8')

            # 4. الكشف عن النص داخل الفقاعة المقتطعة بعتبة ثقة 25%
            text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_URL, ROBOFLOW_API_KEY, b64_crop, confidence=25)
            
            # 5. تطبيق التبييض على النص المكتشف
            if text_predictions:
                for text_pred in text_predictions:
                    text_points = text_pred.get("points", [])
                    if len(text_points) < 3: continue

                    # تحويل الإحداثيات إلى إحداثيات الصورة الأصلية (الـ chunk)
                    text_polygon_chunk_coords = np.array([[int(p["x"]) + x_min_exp, int(p["y"]) + y_min_exp] 
                                                        for p in text_points], dtype=np.int32)
                    
                    # التحقق من بياض الخلفية باستخدام المنطقة الموسعة (25%)
                    if is_background_white(image_chunk, text_polygon_chunk_coords):
                        # نستخدم المضلع الأصلي للنص (text_polygon_chunk_coords) لعملية التبييض نفسها
                        cv2.fillPoly(result_chunk, [text_polygon_chunk_coords], (255, 255, 255))
                        whitened_count += 1
            
        print(f"   Chunk {chunk_index + 1} finished. Whitened {whitened_count} text areas.")
        
        emit_progress(3, f"اكتمل تبييض النصوص في الجزء {chunk_index + 1}...", 
                      50 + int((chunk_index / total_chunks) * 40), sid)
        
        return result_chunk

    except Exception as e:
        print(f"⚠️ Error processing chunk {chunk_index}: {e}")
        return image_chunk

# --- Main Processing Task (كما هو منطق التقسيم) ---
def process_image_task(image_path, output_filename_base, mode, sid):
    # ... (باقي الدالة لم يتم تعديلها لأن التغيير كان في دالة is_background_white وتوابعها) ...
    print(f"ℹ️ SID {sid}: Starting task for {os.path.basename(image_path)}")
    start_time = time.time()
    final_output_path = ""
    result_data = {}

    try:
        emit_progress(0, "تحميل الصورة...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path}")
        
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) 

        h_full, w_full = image.shape[:2]
        
        # --- منطق التقسيم ---
        if h_full > MAX_SPLIT_HEIGHT:
            print(f"ℹ️ Image height ({h_full}px) exceeds limit ({MAX_SPLIT_HEIGHT}px). Splitting...")
            emit_progress(1, "الصورة طويلة جداً، جاري تقسيمها ومعالجتها...", 10, sid)
            
            y_start_2 = MAX_SPLIT_HEIGHT
            
            chunk1 = image[0:y_start_2 + SPLIT_OVERLAP, :]
            chunk2 = image[y_start_2 - SPLIT_OVERLAP:h_full, :]
            
            processed_chunk1 = process_single_chunk(chunk1, 0, 0, 2, sid)
            processed_chunk2 = process_single_chunk(chunk2, y_start_2 - SPLIT_OVERLAP, 1, 2, sid)
            
            print("ℹ️ Stitching results back together...")
            emit_progress(4, "دمج الأجزاء...", 90, sid)
            
            processed_chunk1_cropped = processed_chunk1[0:y_start_2, :]
            processed_chunk2_cropped = processed_chunk2[SPLIT_OVERLAP:, :]
            
            final_image = cv2.vconcat([processed_chunk1_cropped, processed_chunk2_cropped])
            
        else:
            final_image = process_single_chunk(image, 0, 0, 1, sid)

        # Save Result
        output_filename = f"{output_filename_base}_cleaned.jpg" 
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        emit_progress(5, "حفظ الصورة النهائية (JPEG)...", 95, sid)
        cv2.imwrite(final_output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95]) 
        
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Complete {processing_time:.2f}s.")
        emit_progress(6, "تمت المعالجة!", 100, sid)
        
        result_data = {
            'mode': 'clean_white', 
            'imageUrl': f'/results/{output_filename}',
            'original_filename': os.path.basename(image_path)
        }
        socketio.emit('processing_complete', result_data, room=sid)

    except Exception as e:
        print(f"❌❌❌ SID {sid}: FATAL ERROR: {e}")
        traceback.print_exc()
        error_msg = str(e).split('\n')[0]
        emit_error(f"خطأ في المعالجة: {error_msg}", sid)
    finally:
        try:
            if image_path and os.path.exists(image_path):
                if 'temp_zip_' not in image_path:
                    os.remove(image_path)
        except: pass

# --- Routes and Socket Events (بدون تغيير) ---
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
