# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
# --- استيرادات جديدة لملفات ZIP ---
import zipfile 
import io 
import glob
# -----------------------------------
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename
# 🛑 تم حذف باقي الاستيرادات غير ذات الصلة (Gemini, text_formatter, shapely, etc.)

eventlet.monkey_patch()
load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
# ℹ️ تم إضافة 'zip' إلى الامتدادات المسموح بها في الواجهة الأمامية، لكن الخادم هنا يركز على الصور
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'} 
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # زيادة الحد لملفات ZIP الكبيرة (100 MB)

# --- SocketIO Setup (بدون تغيير) ---
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,
    ping_interval=25
)

# --- Ensure directories exist (بدون تغيير) ---
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Directories verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR creating directories: {e}")
    sys.exit(1)

# 🛑 تم حذف جميع الوظائف المتعلقة بالترجمة والرسم (Gemini, find_optimal_text, draw_text)

# --- Helper Functions (بدون تغيير) ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)
def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60):
    # ... (نفس دالة Roboflow) ...
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown Model"
    print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    try:
        response = requests.post(f"{endpoint_url}?api_key={api_key}", data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status(); data = response.json(); predictions = data.get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err: raise ConnectionError(f"Roboflow API ({model_name}) timed out.") from timeout_err
    except requests.exceptions.HTTPError as http_err: raise ConnectionError(f"Roboflow API ({model_name}) failed (Status {http_err.response.status_code}).") from http_err
    except requests.exceptions.RequestException as req_err: raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e: raise RuntimeError(f"Unexpected error during Roboflow ({model_name}).") from e


# --- Main Processing Task (Text Cleaning Only) ---
def process_image_task(image_path, output_filename_base, sid, original_filename=None): # تمت إضافة original_filename
    print(f"ℹ️ SID {sid}: Starting image cleaning task for {image_path}")
    start_time = time.time(); final_image_np = None; result_data = {}
    final_output_path = ""
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    
    # تحديد اسم الملف الأصلي للنتيجة (مهم لوضع الدفعة)
    if original_filename is None: 
        original_filename = os.path.basename(image_path)
    
    try:
        emit_progress(0, f"Loading image: {original_filename}...", 5, sid);
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path} (Task Level).")
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        result_image = image.copy(); text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 1. اكتشاف النص
        emit_progress(1, f"Detecting text in {original_filename}...", 10, sid);
        retval, buffer_text = cv2.imencode('.jpg', image);
        if not retval or buffer_text is None: raise ValueError("Failed encode text detect.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8'); text_predictions = []
        
        try:
            text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image_text)
            polygons_drawn = 0
            
            # 2. إنشاء قناع
            for pred in text_predictions:
                 points = pred.get("points", []);
                 if len(points) >= 3:
                     polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                     polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                     try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                     except Exception: print(f"⚠️ Warn: Error drawing text polygon for {original_filename}")
            
            # 3. التنظيف (Inpainting)
            if np.any(text_mask):
                 emit_progress(2, f"Cleaning text in {original_filename}...", 50, sid);
                 inpainted_image_cv = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS)
                 if inpainted_image_cv is None: raise RuntimeError("cv2.inpaint returned None")
                 result_image = inpainted_image_cv
            else: 
                 emit_progress(2, f"No text found in {original_filename}.", 50, sid);
                 
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: 
            print(f"❌ SID {sid}: Error during Roboflow text detection for {original_filename}: {rf_err}. Saving original image.")
            emit_error(f"Text detection failed for {original_filename}.", sid)
            result_image = image
        except Exception as e: 
            print(f"❌ SID {sid}: Error during text detection/inpainting for {original_filename}: {e}. Saving original image.")
            emit_error(f"Processing error for {original_filename}.", sid)
            result_image = image
        
        # 4. حفظ الصورة النظيفة
        emit_progress(3, f"Saving cleaned image: {original_filename}...", 90, sid);
        final_image_np = result_image
        output_filename = f"{output_filename_base}_cleaned.jpg" 
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        # ℹ️ تمرير اسم الملف الأصلي لإعادته إلى الواجهة الأمامية (مهم لوضع الدفعة)
        result_data = {'mode': 'clean', 'imageUrl': f'/results/{output_filename}', 'original_filename': original_filename} 

        if final_image_np is not None and final_output_path:
            save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np);
                 if not save_success: raise IOError(f"cv2.imwrite failed: {final_output_path}")
            except Exception as cv_save_err:
                 # محاولة الحفظ باستخدام PIL كخطة بديلة
                 from PIL import Image
                 pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB)); 
                 os.makedirs(os.path.dirname(final_output_path), exist_ok=True); 
                 pil_img_to_save.save(final_output_path); 
                 save_success = True
            
            if save_success:
                processing_time = time.time() - start_time; 
                emit_progress(4, f"Finished {original_filename} ({processing_time:.2f}s).", 100, sid)
                socketio.emit('processing_complete', result_data, room=sid)
            else: 
                emit_error(f"Could not save final image for {original_filename}.", sid)
    
    except Exception as e:
        print(f"❌❌❌ SID {sid}: UNHANDLED FATAL ERROR in task for {original_filename}: {e}")
        traceback.print_exc()
        emit_error(f"Unexpected server error during cleaning of {original_filename}.", sid)
    
    finally:
        # يتم حذف الملف المستخرج بعد المعالجة، لكن ليس الملف المضغوط نفسه
        try:
            if image_path and os.path.exists(image_path) and not image_path.endswith('.zip'): 
                os.remove(image_path)
                print(f"🧹 SID {sid}: Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ SID {sid}: Error cleaning up {image_path}: {cleanup_err}")

# --- Flask Routes ---
@app.route('/upload', methods=['POST'])
def handle_upload():
    # ... (نفس دالة handle_upload للصورة الفردية) ...
    temp_log_id = f"upload_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST upload request ---")
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    if '.' not in filename: return jsonify({'error': 'File has no extension'}), 400
    ext = filename.rsplit('.', 1)[1].lower()
    
    # ℹ️ السماح بملفات ZIP هنا لتوحيد عملية الرفع، لكننا سنفصلها في المسار التالي /upload_zip
    if ext not in ALLOWED_EXTENSIONS and ext != 'zip':
        return jsonify({'error': f'Invalid file type: {ext}. Allowed: {", ".join(ALLOWED_EXTENSIONS)} or zip'}), 400
    
    unique_id = uuid.uuid4()
    input_filename = f"{unique_id}.{ext}"
    output_filename_base = f"{unique_id}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    
    # إذا تم رفع ملف ZIP عن طريق الخطأ إلى /upload، يجب رفضه
    if ext == 'zip':
        print(f"[{temp_log_id}] ❌ Upload Error: ZIP file uploaded to single endpoint.")
        return jsonify({'error': 'Please use the correct endpoint for ZIP files.'}), 400
        
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(upload_path)
        print(f"[{temp_log_id}] ✔️ File saved via POST: {upload_path}")
        return jsonify({
            'message': 'File uploaded successfully',
            'output_filename_base': output_filename_base,
            'saved_filename': input_filename
        }), 200
    except Exception as e:
        if os.path.exists(upload_path): os.remove(upload_path)
        return jsonify({'error': 'Unexpected server error during upload'}), 500

@app.route('/upload_zip', methods=['POST'])
def handle_upload_zip():
    temp_log_id = f"upload_zip_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST ZIP upload request ---")
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.lower().endswith('.zip'):
        return jsonify({'error': 'Only ZIP files are allowed on this endpoint'}), 400
        
    unique_upload_dir_name = f"zip_{uuid.uuid4()}"
    extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], unique_upload_dir_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    images_to_process = []
    
    try:
        with zipfile.ZipFile(io.BytesIO(file.read()), 'r') as zip_ref:
            # التحقق من الملفات القابلة للمعالجة قبل الاستخراج
            valid_files = [
                f for f in zip_ref.namelist() 
                if not f.startswith('__MACOSX/') and f.split('.')[-1].lower() in ALLOWED_EXTENSIONS
            ]
            
            if not valid_files:
                return jsonify({'error': 'No valid image files (PNG, JPG, WEBP) found in the ZIP.'}), 400
                
            # استخراج الملفات الصالحة فقط إلى المجلد المؤقت
            zip_ref.extractall(extract_dir, members=valid_files)
            print(f"[{temp_log_id}] ✔️ Extracted {len(valid_files)} valid images to {extract_dir}")

            # تجميع أسماء الملفات التي سيتم معالجتها
            for filename in valid_files:
                # يفضل استخدام os.path.join لضمان المسارات الصحيحة
                full_path = os.path.join(extract_dir, filename)
                # التأكد من أن الملف ليس مجلدًا وأنه موجود
                if os.path.isfile(full_path):
                    # نستخدم اسم الملف كقاعدة للناتج المؤقت
                    base_name = os.path.basename(filename)
                    output_base = f"{uuid.uuid4()}" # قاعدة اسم ناتج فريدة لكل صورة
                    
                    images_to_process.append({
                        'input_path': full_path,
                        'output_filename_base': output_base,
                        'original_filename': base_name
                    })

        # إرجاع قائمة الصور للمعالجة و مسار المجلد المؤقت للتنظيف لاحقاً
        return jsonify({
            'message': f'ZIP uploaded and {len(images_to_process)} files ready.',
            'images_to_process': images_to_process,
            'temp_dir': extract_dir # لإزالة المجلد بعد اكتمال الدفعة
        }), 200
        
    except Exception as e:
        print(f"[{temp_log_id}] ❌ ZIP Error: {e}")
        # التنظيف في حالة الفشل
        try: os.rmdir(extract_dir)
        except OSError: pass # قد يكون هناك ملفات، لذلك نستدعي دالة التنظيف
        return jsonify({'error': f'Server error processing ZIP file: {e}'}), 500

# --- SocketIO Event Handlers ---
# ... (handle_connect و handle_disconnect بدون تغيير) ...
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    # ... (نفس دالة handle_start_processing للصورة الفردية) ...
    sid = request.sid
    output_filename_base = data.get('output_filename_base')
    saved_filename = data.get('saved_filename')
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(saved_filename))
    if not os.path.exists(upload_path):
        emit_error(f"Uploaded file '{saved_filename}' not found on server.", sid)
        return
        
    try:
        socketio.start_background_task(
            process_image_task,
            image_path=upload_path,
            output_filename_base=output_filename_base,
            sid=sid,
            original_filename=saved_filename # نمرر الاسم الأصلي لمرة واحدة
        )
        socketio.emit('processing_started', {'message': 'File received, cleaning started...'}, room=sid)
    except Exception as task_start_err:
        emit_error(f"Server error starting image processing task.", sid)

@socketio.on('start_batch_processing')
def handle_start_batch_processing(data):
    sid = request.sid
    images_to_process = data.get('images_to_process', [])
    temp_dir = data.get('temp_dir') # المسار المؤقت الذي أنشأناه في /upload_zip
    
    if not images_to_process or not temp_dir:
        emit_error("Missing batch data or temporary directory.", sid)
        return

    print(f"ℹ️ SID {sid}: Starting batch process for {len(images_to_process)} images in {temp_dir}")
    socketio.emit('batch_started', {'total_images': len(images_to_process)}, room=sid)

    # يمكننا إضافة دالة تنظيف المجلد هنا، أو الاعتماد على دالة منفصلة
    def cleanup_temp_dir(dir_path):
        import shutil
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"🧹 Cleaned up temporary batch directory: {dir_path}")
        except Exception as cleanup_err:
            print(f"⚠️ Error cleaning up batch directory {dir_path}: {cleanup_err}")

    # ℹ️ نستخدم عملية خلفية واحدة لتشغيل جميع مهام المعالجة بشكل متسلسل أو متزامن (حسب eventlet)
    # هنا سنشغلها بشكل متسلسل لتبسيط Log والتعامل مع موارد CPU/Roboflow
    def batch_worker(images_list, sid):
        for i, img_data in enumerate(images_list):
            # تحديث التقدم الكلي للدفعة
            overall_percentage = int((i / len(images_list)) * 100)
            emit_progress(-1, f"Batch: Processing {i + 1}/{len(images_list)}: {img_data['original_filename']}", overall_percentage, sid)
            
            try:
                # نستخدم process_image_task لمعالجة الصورة
                # بما أننا داخل مهمة خلفية، يمكننا استدعاء الوظيفة مباشرة
                process_image_task(
                    image_path=img_data['input_path'],
                    output_filename_base=img_data['output_filename_base'],
                    sid=sid,
                    original_filename=img_data['original_filename']
                )
            except Exception as e:
                print(f"❌ Batch Item Error: {img_data['original_filename']}: {e}")
                emit_error(f"Error processing {img_data['original_filename']}.", sid)
                
        # إرسال إشارة اكتمال الدفعة الكلية
        emit_progress(-1, "Batch processing complete.", 100, sid)
        # التنظيف بعد انتهاء المعالجة
        cleanup_temp_dir(temp_dir)
        socketio.emit('batch_complete', {'message': 'All images processed and temporary files cleaned.'}, room=sid)


    try:
        # تشغيل العامل في مهمة خلفية منفصلة
        socketio.start_background_task(batch_worker, images_to_process, sid)
    except Exception as task_start_err:
        emit_error("Server error starting batch worker.", sid)
        cleanup_temp_dir(temp_dir)

# ... (باقي المسارات والإعدادات بدون تغيير) ...

# --- Main Execution (بدون تغيير) ---
if __name__ == '__main__':
    # ... (تشغيل التطبيق) ...
    # ℹ️ تأكد من أنك تستخدم هذا الكود المحدَّث بالكامل
    print("--- Starting Manga Processor Web App (Cleaner Mode with ZIP support) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! Text detection will fail. ███")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Starting server http://0.0.0.0:{port}")
    try:
        # يجب استيراد PIL هنا فقط لتجنب الأخطاء في وظيفة Save fallback
        from PIL import Image # لضمان توفرها إذا لزم الأمر في خطة الحفظ البديلة
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}")
        sys.exit(1)

