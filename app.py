# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename

# 🛑 تمت إزالة جميع استيرادات ومعالجة مكتبات PIL, Image, ImageDraw, ImageFont, re, shapely, google.generativeai, io
# 🛑 تمت إزالة فئة DummyTextFormatter و text_formatter

eventlet.monkey_patch()
load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
# ℹ️ نحتفظ بمفتاح Roboflow فقط لاكتشاف النص
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# 🛑 تمت إزالة GOOGLE_API_KEY

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# --- SocketIO Setup ---
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# --- Ensure directories exist ---
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Directories verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR creating directories: {e}")
    sys.exit(1)

# 🛑 تمت إزالة setup_font

# --- Constants ---
# 🛑 تمت إزالة ثوابت الخطوط والنصوص
# 🛑 تمت إزالة PROMPT_GEMINI

# 🛑 تمت إزالة تهيئة Google GenAI

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)
def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

# --- Core Functions (Simplified) ---
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60):
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown Model"
    print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    try:
        response = requests.post(f"{endpoint_url}?api_key={api_key}", data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status(); data = response.json(); predictions = data.get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err: print(f"❌ Roboflow ({model_name}) Timeout: {timeout_err}"); raise ConnectionError(f"Roboflow API ({model_name}) timed out.") from timeout_err
    except requests.exceptions.HTTPError as http_err: print(f"❌ Roboflow ({model_name}) HTTP Error: Status {http_err.response.status_code}"); print(f"   Response: {http_err.response.text[:200]}"); raise ConnectionError(f"Roboflow API ({model_name}) failed (Status {http_err.response.status_code}).") from http_err
    except requests.exceptions.RequestException as req_err: print(f"❌ Roboflow ({model_name}) Request Error: {req_err}"); raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e: print(f"❌ Roboflow ({model_name}) Unexpected Error: {e}"); traceback.print_exc(limit=2); raise RuntimeError(f"Unexpected error during Roboflow ({model_name}).") from e

# 🛑 تمت إزالة extract_translation
# 🛑 تمت إزالة ask_gemini_translation
# 🛑 تمت إزالة find_optimal_text_settings_final
# 🛑 تمت إزالة draw_text_on_layer

# --- Main Processing Task (Simplified) ---
def process_image_task(image_path, output_filename_base, sid): # 🛑 تمت إزالة 'mode'
    print(f"ℹ️ SID {sid}: Starting image cleaning task for {image_path}")
    start_time = time.time(); final_image_np = None; result_data = {}
    final_output_path = ""
    # ℹ️ نحتاج فقط إلى نموذج اكتشاف النص
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    # 🛑 تمت إزالة ROBOFLOW_BUBBLE_DETECT_URL
    try:
        emit_progress(0, "Loading image...", 5, sid);
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path} (Task Level).")
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        # ℹ️ سنستخدم result_image كمتغير للعملية الناتجة بعد التنظيف
        result_image = image.copy(); text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 1. اكتشاف النص
        emit_progress(1, "Detecting text...", 10, sid);
        retval, buffer_text = cv2.imencode('.jpg', image);
        if not retval or buffer_text is None: raise ValueError("Failed encode text detect.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8'); text_predictions = []
        
        try:
            print(f"   Sending request to Roboflow text detection...")
            text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image_text)
            print(f"   Found {len(text_predictions)} potential text areas.")
            polygons_drawn = 0
            
            # 2. إنشاء قناع
            for pred in text_predictions:
                 points = pred.get("points", []);
                 if len(points) >= 3:
                     polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                     polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                     try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                     except Exception as fill_err: print(f"⚠️ Warn: Error drawing text polygon: {fill_err}")
            print(f"   Created text mask with {polygons_drawn} polygons.")

            # 3. التنظيف (Inpainting)
            if np.any(text_mask):
                 emit_progress(2, "Inpainting (cleaning) text areas...", 50, sid);
                 print(f"   Inpainting detected text areas...")
                 inpainted_image_cv = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS)
                 if inpainted_image_cv is None: raise RuntimeError("cv2.inpaint returned None")
                 result_image = inpainted_image_cv
                 print(f"   Inpainting complete.")
            else: 
                 emit_progress(2, "No text detected, saving original...", 50, sid);
                 print(f"   No text detected, skipping inpainting.")

        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: 
            print(f"❌ SID {sid}: Error during Roboflow text detection: {rf_err}. Saving original image."); 
            emit_error("Text detection failed. Saving original image.", sid)
            result_image = image # إذا فشل الكشف، نستخدم الصورة الأصلية

        except Exception as e: 
            print(f"❌ SID {sid}: Error during text detection/inpainting: {e}. Saving original image."); 
            emit_error("Text processing error. Saving original image.", sid)
            result_image = image # إذا فشل التنظيف، نستخدم الصورة الأصلية
        
        # 4. حفظ الصورة النظيفة
        emit_progress(3, "Finalizing and saving image...", 90, sid);
        final_image_np = result_image
        output_filename = f"{output_filename_base}_cleaned.jpg" # تسمية الناتج
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        result_data = {'mode': 'clean', 'imageUrl': f'/results/{output_filename}'} # ℹ️ تغيير وضع النتيجة

        if final_image_np is not None and final_output_path:
            save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np);
                 if not save_success: raise IOError(f"cv2.imwrite failed: {final_output_path}")
                 print(f"✔️ Saved (OpenCV): {final_output_path}")
            except Exception as cv_save_err:
                 print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL...");
                 try: # محاولة الحفظ باستخدام PIL كخطة بديلة
                     from PIL import Image
                     pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB)); 
                     os.makedirs(os.path.dirname(final_output_path), exist_ok=True); 
                     pil_img_to_save.save(final_output_path); 
                     save_success = True; print(f"✔️ Saved (PIL): {final_output_path}")
                 except Exception as pil_save_err: 
                     print(f"❌ PIL save failed: {pil_save_err}"); 
                     emit_error("Failed save final image.", sid)
            
            if save_success:
                processing_time = time.time() - start_time; print(f"✔️ SID {sid} Complete {processing_time:.2f}s."); 
                emit_progress(4, f"Complete ({processing_time:.2f}s).", 100, sid)
                socketio.emit('processing_complete', result_data, room=sid)
            else: 
                print(f"❌❌❌ SID {sid}: Critical Error: Could not save image {final_output_path}")
        else: 
            print(f"❌ SID {sid}: No final image data."); emit_error("Internal error: No final image.", sid)
    
    except Exception as e:
        print(f"❌❌❌ SID {sid}: UNHANDLED FATAL ERROR in task: {e}")
        traceback.print_exc()
        emit_error(f"Unexpected server error during cleaning ({type(e).__name__}).", sid)
    
    finally:
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 SID {sid}: Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ SID {sid}: Error cleaning up {image_path}: {cleanup_err}")

# --- Flask Routes ---
@app.route('/')
def index():
    # ℹ️ يجب تعديل ملف index.html لتناسب وضع التنظيف فقط
    return render_template('index.html')

@app.route('/results/<path:filename>')
def get_result_image(filename):
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    results_dir = os.path.abspath(app.config['RESULT_FOLDER'])
    try:
        return send_from_directory(results_dir, filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

@app.route('/upload', methods=['POST'])
def handle_upload():
    temp_log_id = f"upload_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST upload request ---")
    if 'file' not in request.files:
        print(f"[{temp_log_id}] ❌ Upload Error: No file part")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        print(f"[{temp_log_id}] ❌ Upload Error: No selected file")
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            print(f"[{temp_log_id}] ❌ Upload Error: Invalid file type '{ext}'")
            return jsonify({'error': f'Invalid file type: {ext}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    else:
         print(f"[{temp_log_id}] ❌ Upload Error: File has no extension")
         return jsonify({'error': 'File has no extension'}), 400
    unique_id = uuid.uuid4()
    input_filename = f"{unique_id}.{ext}"
    output_filename_base = f"{unique_id}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(upload_path)
        file_size_kb = os.path.getsize(upload_path) / 1024
        print(f"[{temp_log_id}] ✔️ File saved via POST: {upload_path} ({file_size_kb:.1f} KB)")
        return jsonify({
            'message': 'File uploaded successfully',
            'output_filename_base': output_filename_base,
            'saved_filename': input_filename
        }), 200
    except IOError as io_err:
        print(f"[{temp_log_id}] ❌ Upload Error: File write error: {io_err}")
        if os.path.exists(upload_path):
            try: os.remove(upload_path); print(f"[{temp_log_id}] 🧹 Cleaned up partial file after IO error.")
            except Exception: pass
        return jsonify({'error': 'Server error saving file'}), 500
    except Exception as e:
        print(f"[{temp_log_id}] ❌ Upload Error: Unexpected error during save: {e}")
        traceback.print_exc()
        if os.path.exists(upload_path):
             try: os.remove(upload_path); print(f"[{temp_log_id}] 🧹 Cleaned up file after unexpected save error.")
             except Exception: pass
        return jsonify({'error': 'Unexpected server error during upload'}), 500

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    sid = request.sid
    print(f"\n--- Received 'start_processing' event SID: {sid} ---")
    if not isinstance(data, dict):
        emit_error("Invalid request data.", sid); return
    output_filename_base = data.get('output_filename_base')
    saved_filename = data.get('saved_filename')
    # 🛑 تمت إزالة 'mode' من التحقق والتعامل
    print(f"   Data received: {data}")
    if not output_filename_base:
        emit_error('Missing file identifier (output_filename_base).', sid); return
    if not saved_filename:
        emit_error('Missing saved filename.', sid); return
        
    upload_dir = app.config['UPLOAD_FOLDER']
    upload_path = os.path.join(upload_dir, secure_filename(saved_filename))
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        print(f"❌ ERROR SID {sid}: File not found at path deduced from event: {upload_path}")
        emit_error(f"Uploaded file '{saved_filename}' not found on server. Please try uploading again.", sid)
        return
    print(f"   File confirmed exists: {upload_path}")
    print(f"   Attempting to start background cleaning task...")
    try:
        socketio.start_background_task(
            process_image_task,
            image_path=upload_path,
            output_filename_base=output_filename_base,
            sid=sid
        )
        print(f"   ✔️ Task initiated SID {sid} for {saved_filename}.")
        socketio.emit('processing_started', {'message': 'File received, cleaning started...'}, room=sid)
    except Exception as task_start_err:
        print(f"❌ CRITICAL SID {sid}: Failed to start background task: {task_start_err}")
        traceback.print_exc()
        emit_error(f"Server error starting image cleaning task.", sid)
    print(f"--- Finished handle 'start_processing' SID: {sid} ---")

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App (Cleaner Mode) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! Text detection will fail. ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default Flask SECRET_KEY!")
    # 🛑 تمت إزالة تحذير GOOGLE_API_KEY
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Ext: {ALLOWED_EXTENSIONS}")
    # 🛑 تمت إزالة معلومات Text Formatter
    print(f"   * Roboflow Key: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Mode: Text Cleaning (Inpainting) Only")
    print(f"   * Starting server http://0.0.0.0:{port}")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}")
        sys.exit(1)
