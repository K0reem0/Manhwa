# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
from PIL import Image, ImageDraw, ImageFont
from requests.exceptions import RequestException
import re
# Removed shapely imports
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename
# Removed google.generativeai imports
import io
import zipfile
import shutil

eventlet.monkey_patch()

# --- Define DummyTextFormatter Globally First (Kept for compatibility but functions are unused) ---
class DummyTextFormatter:
    def __init__(self): print("⚠️ WARNING: Initializing DummyTextFormatter.")
    def set_arabic_font_path(self, path): pass
    def get_font(self, size): return None
    def format_arabic_text(self, text): return text
    def layout_balanced_text(self, draw, text, font, target_width): return text

try:
    # Removed text_formatter import logic as it's no longer strictly needed but keeping structure
    # If the file 'text_formatter.py' is missing, the Dummy will be used anyway.
    # To keep the original structure for non-fatal imports:
    import text_formatter
    print("✔️ Successfully imported 'text_formatter.py'.")
    if not all(hasattr(text_formatter, func) for func in ['set_arabic_font_path', 'get_font', 'format_arabic_text', 'layout_balanced_text']):
         print("⚠️ WARNING: 'text_formatter.py' missing required functions!"); raise ImportError("Missing functions")
except ImportError as import_err:
    print(f"❌ ERROR: Cannot import 'text_formatter.py': {import_err}")
    text_formatter = DummyTextFormatter()
    print("⚠️ WARNING: Using dummy 'text_formatter'.")


load_dotenv()
# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_ARCHIVES = {'zip'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# Removed GOOGLE_API_KEY
# Removed GenAI-related constants

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # Increased limit for zip files (50 MB)

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

# --- Font Setup (Kept for completeness but unused) ---
def setup_font():
    font_path_to_set = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path_rel = os.path.join(script_dir, "fonts", "66Hayah.otf")
        potential_path_cwd = os.path.join(".", "fonts", "66Hayah.otf")
        if os.path.exists(potential_path_rel): font_path_to_set = potential_path_rel
        elif os.path.exists(potential_path_cwd): font_path_to_set = potential_path_cwd
        if font_path_to_set:
            print(f"ℹ️ Font found: '{font_path_to_set}'. Attempting to set path.")
            try: text_formatter.set_arabic_font_path(font_path_to_set)
            except AttributeError: print("   (Formatter missing method? Skipping.)")
        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found. Using default.")
            try: text_formatter.set_arabic_font_path(None)
            except AttributeError: print("   (Formatter missing method? Skipping.)")
    except Exception as e:
        print(f"❌ Error finding font path: {e}. Using default.")
        try: text_formatter.set_arabic_font_path(None)
        except AttributeError: print("   (Formatter missing method? Skipping.)")
        except Exception as E2: print(f"❌ Error setting font path to None: {E2}")
setup_font()

# --- Constants ---
# Removed text color/shadow constants
# Removed TRANSLATION_PROMPT_GEMINI

# --- Initialize Google GenAI (Removed) ---
# model = None ... (Removed all Gemini initialization code)


# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    """Emits progress updates to a specific client."""
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)
def emit_error(message, sid):
    """Emits an error message to a specific client."""
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)


# --- Core Functions ---
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60):
    """Fetches predictions from a Roboflow model."""
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

# Removed extract_translation function
# Removed ask_gemini_translation function
# Removed find_optimal_text_settings_final function
# Removed draw_text_on_layer function

# --- Main Processing Task (MODIFIED) ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """
    Main background task to process a single image.
    This function is now modified to only perform text detection and white inpainting.
    """
    print(f"ℹ️ SID {sid}: Starting image processing task for {os.path.basename(image_path)}")
    start_time = time.time(); final_image_np = None; result_data = {}; image = None
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    # Removed ROBOFLOW_BUBBLE_DETECT_URL

    # The 'mode' parameter is now ignored but kept in the function signature for compatibility with calling functions.
    
    try:
        emit_progress(0, "Loading image...", 5, sid);
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path} (Task Level).")
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        result_image = image.copy(); text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        emit_progress(1, "Detecting text...", 10, sid);
        retval, buffer_text = cv2.imencode('.jpg', image);
        if not retval or buffer_text is None: raise ValueError("Failed encode text detect.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8'); text_predictions = []
        
        try:
            print(f"   Sending request to Roboflow text detection...")
            text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image_text)
            print(f"   Found {len(text_predictions)} potential text areas.")
            polygons_drawn = 0
            for pred in text_predictions:
                 points = pred.get("points", []);
                 if len(points) >= 3:
                     polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                     polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                     try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                     except Exception as fill_err: print(f"⚠️ Warn: Error drawing text polygon: {fill_err}")
            print(f"   Created text mask with {polygons_drawn} polygons.")
            
            if np.any(text_mask):
                 print(f"   Filling detected text areas with WHITE color...")
                 
                 # --- MODIFICATION: Inpainting with white color instead of cv2.inpaint ---
                 # Create an inverted mask where the text areas are 0 and everything else is 1
                 # Note: text_mask is 255 for text, 0 for background.
                 # We want to keep the original image where text_mask is 0 (background)
                 # and set it to white (255, 255, 255) where text_mask is 255 (text area).
                 
                 # Convert the 1-channel mask to a 3-channel mask for BGR image
                 mask_3ch = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
                 
                 # Create a white canvas of the same size as the image
                 white_canvas = np.full_like(image, 255, dtype=np.uint8)
                 
                 # Use the mask to blend: 
                 # 1. Where text_mask is 255 (text area), take the WHITE_CANVAS (i.e., fill with white)
                 # 2. Where text_mask is 0 (background), take the ORIGINAL IMAGE
                 
                 # result_image = (original_image * (1 - mask_ratio)) + (white_canvas * mask_ratio)
                 # A simpler way with array operations (mask is 0 or 255):
                 # result_image = cv2.bitwise_not(mask_3ch) & image  # Keeps image outside the mask
                 # result_image = result_image + (mask_3ch & white_canvas) # Adds white inside the mask
                 
                 # A clearer way: use the mask to overwrite text areas with white
                 result_image[text_mask > 0] = 255 # Sets the pixel values in the text area to white (255, 255, 255 BGR)
                 final_image_np = result_image
                 print(f"   White filling complete.")
            else: 
                print(f"   No text detected, saving original image.")
                final_image_np = result_image
                
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: 
            print(f"❌ SID {sid}: Error during Roboflow text detection: {rf_err}. Saving original image."); 
            emit_error("Text detection failed. Saving original.", sid);
            final_image_np = result_image
        except Exception as e: 
            print(f"❌ SID {sid}: Error during text detection/inpainting: {e}. Saving original image."); 
            emit_error("Text processing error. Saving original.", sid);
            final_image_np = result_image

        # --- Remaining Logic: Saving the final image ---
        # The logic below assumes no bubble processing, just text removal (white filling)
        
        emit_progress(2, "Finalizing image...", 90, sid);
        
        output_filename = f"{output_filename_base}_cleaned_white.jpg"
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        
        if final_image_np is not None and final_output_path:
            emit_progress(5, "Saving final image...", 98, sid); save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np);
                 if not save_success: raise IOError(f"cv2.imwrite failed: {final_output_path}")
                 print(f"✔️ Saved (OpenCV): {final_output_path}")
            except Exception as cv_save_err:
                 print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL...");
                 try: 
                     pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB)); 
                     os.makedirs(os.path.dirname(final_output_path), exist_ok=True); 
                     pil_img_to_save.save(final_output_path); 
                     save_success = True; print(f"✔️ Saved (PIL): {final_output_path}")
                 except Exception as pil_save_err: 
                     print(f"❌ PIL save failed: {pil_save_err}"); 
                     emit_error("Failed save final image.", sid)
            
            if save_success:
                processing_time = time.time() - start_time;
                print(f"✔️ SID {sid} Complete {processing_time:.2f}s for {os.path.basename(image_path)}.");
                emit_progress(6, f"Complete ({processing_time:.2f}s).", 100, sid)
                
                # Simplified result data
                result_data = {
                    'mode': 'white_fill', # New mode name to reflect change
                    'imageUrl': f'/results/{os.path.basename(final_output_path)}',
                    'original_filename': os.path.basename(image_path),
                    'is_zip_batch': False
                }
                socketio.emit('processing_complete', result_data, room=sid)
            else:
                print(f"❌❌❌ SID {sid}: Critical Error: Could not save image {final_output_path}")
        elif not final_output_path:
            print(f"❌ SID {sid}: Aborted before output path set.")
        else:
            print(f"❌ SID {sid}: No final image data."); emit_error("Internal error: No final image.", sid)

    except Exception as e:
        print(f"❌❌❌ SID {sid}: UNHANDLED FATAL ERROR in task: {e}")
        traceback.print_exc()
        emit_error(f"Unexpected server error during processing ({type(e).__name__}).", sid)
    finally:
        # Cleanup the uploaded file for both single and batch processing
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 SID {sid}: Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ SID {sid}: Error cleaning up {image_path}: {cleanup_err}")

# --- Flask Routes (No changes needed) ---
@app.route('/')
def index():
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

# --- MODIFIED: Single file upload route ---
@app.route('/upload', methods=['POST'])
def handle_upload():
    """
    Handles single image file uploads.
    """
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

# --- Zip file upload route (No changes needed) ---
@app.route('/upload_zip', methods=['POST'])
def handle_zip_upload():
    """
    Handles zip file uploads, extracts images, and prepares them for processing.
    """
    temp_log_id = f"upload_zip_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST zip upload request ---")
    if 'file' not in request.files:
        print(f"[{temp_log_id}] ❌ Upload Error: No file part")
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        print(f"[{temp_log_id}] ❌ Upload Error: No selected file")
        return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.zip'):
        print(f"[{temp_log_id}] ❌ Upload Error: Invalid file type. Expected .zip")
        return jsonify({'error': 'Invalid file type. Please upload a .zip file.'}), 400

    temp_dir_for_extraction = None
    try:
        zip_file_bytes = file.read()
        zip_file = zipfile.ZipFile(io.BytesIO(zip_file_bytes))
        extracted_images_info = []
        temp_dir_for_extraction = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_zip_{uuid.uuid4()}")
        os.makedirs(temp_dir_for_extraction, exist_ok=True)
        print(f"[{temp_log_id}] Created temporary directory for extraction: {temp_dir_for_extraction}")

        for file_info in zip_file.infolist():
            if file_info.is_dir(): continue
            filename = os.path.basename(file_info.filename)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            if ext in ALLOWED_EXTENSIONS:
                extracted_file_path = os.path.join(temp_dir_for_extraction, secure_filename(filename))
                try:
                    with zip_file.open(file_info) as source, open(extracted_file_path, "wb") as target:
                        target.write(source.read())
                    print(f"[{temp_log_id}] Extracted image: {extracted_file_path}")
                    extracted_images_info.append({
                        'original_filename': filename,
                        'saved_path': extracted_file_path,
                        'output_base': f"zip_img_{uuid.uuid4()}" # Unique base for output filenames
                    })
                except Exception as extract_err:
                    print(f"[{temp_log_id}] ⚠️ Warning: Could not extract {filename}: {extract_err}")
            else:
                print(f"[{temp_log_id}] Skipping non-image file in zip: {filename}")
        if not extracted_images_info:
            print(f"[{temp_log_id}] ❌ Upload Error: No valid images found in the zip file.")
            if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
            return jsonify({'error': 'No valid images found in the zip file.'}), 400

        print(f"[{temp_log_id}] Successfully extracted {len(extracted_images_info)} images.")
        return jsonify({
            'message': 'Zip file uploaded and images extracted successfully',
            'images_to_process': extracted_images_info
        }), 200
    except zipfile.BadZipFile:
        print(f"[{temp_log_id}] ❌ Upload Error: Bad zip file.")
        if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
        return jsonify({'error': 'The uploaded file is not a valid zip archive.'}), 400
    except IOError as io_err:
        print(f"[{temp_log_id}] ❌ Upload Error: File write error: {io_err}")
        if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
        return jsonify({'error': 'Server error saving zip file or extracted images.'}), 500
    except Exception as e:
        print(f"[{temp_log_id}] ❌ Upload Error: Unexpected error during zip processing: {e}")
        traceback.print_exc()
        if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
        return jsonify({'error': 'Unexpected server error during zip upload processing.'}), 500

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

# --- MODIFIED: Single image processing handler ---
@socketio.on('start_processing')
def handle_start_processing(data):
    """Handles the start of a single-image processing task."""
    sid = request.sid
    print(f"\n--- Received 'start_processing' event SID: {sid} ---")
    if not isinstance(data, dict):
        emit_error("Invalid request data.", sid); return
    output_filename_base = data.get('output_filename_base')
    saved_filename = data.get('saved_filename')
    mode = data.get('mode') # The mode is now ignored in process_image_task but we keep the check
    print(f"   Data received: {data}")
    
    # We allow any mode to pass, but the process_image_task will only do white-fill inpainting now.
    if not output_filename_base or not saved_filename:
        emit_error('Missing or invalid parameters.', sid); return
    print(f"   Mode: '{mode}' (Processing will now only perform white inpainting)")
    
    upload_dir = app.config['UPLOAD_FOLDER']
    upload_path = os.path.join(upload_dir, secure_filename(saved_filename))
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        print(f"❌ ERROR SID {sid}: File not found at path: {upload_path}")
        emit_error(f"Uploaded file '{saved_filename}' not found on server.", sid)
        return
    print(f"   File confirmed exists: {upload_path}")
    try:
        socketio.start_background_task(
            process_image_task,
            image_path=upload_path,
            output_filename_base=output_filename_base,
            mode=mode, # Pass original mode
            sid=sid
        )
        print(f"   ✔️ Task initiated SID {sid} for {saved_filename}.")
        socketio.emit('processing_started', {'message': 'File received, processing started...'}, room=sid)
    except Exception as task_start_err:
        print(f"❌ CRITICAL SID {sid}: Failed to start background task: {task_start_err}")
        traceback.print_exc()
        emit_error(f"Server error starting image processing task.", sid)

# --- NEW: Batch processing handler for zip files (Mode is passed but ignored by task) ---
@socketio.on('start_batch_processing')
def handle_start_batch_processing(data):
    """
    Handles the start of a batch-image processing task for all images from a zip file.
    """
    sid = request.sid
    print(f"\n--- Received 'start_batch_processing' event SID: {sid} ---")
    if not isinstance(data, dict):
        emit_error("Invalid request data.", sid); return
    images_to_process = data.get('images_to_process', [])
    mode = data.get('mode')
    if not isinstance(images_to_process, list) or not images_to_process:
        emit_error('Invalid or empty image list for batch processing.', sid); return

    print(f"   Initiating batch processing for {len(images_to_process)} images. Mode: {mode} (Processing will now only perform white inpainting)")
    socketio.emit('batch_started', {'total_images': len(images_to_process)}, room=sid)

    for img_info in images_to_process:
        try:
            image_path = img_info.get('saved_path')
            output_base = img_info.get('output_base')
            original_filename = img_info.get('original_filename', 'unknown')
            if not image_path or not output_base:
                print(f"   ⚠️ Skipping image {original_filename}: missing path or output base.")
                emit_error(f"Skipping image '{original_filename}' due to missing data.", sid)
                continue
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                print(f"❌ ERROR SID {sid}: Image file not found: {image_path}")
                emit_error(f"Image '{original_filename}' not found on server. Skipping.", sid)
                continue

            # Start a separate background task for each image
            socketio.start_background_task(
                process_image_task,
                image_path=image_path,
                output_filename_base=output_base,
                mode=mode, # Pass original mode
                sid=sid
            )
            print(f"   ✔️ Task initiated for image '{original_filename}'.")
        except Exception as e:
            print(f"❌ CRITICAL SID {sid}: Failed to start task for an image: {e}")
            traceback.print_exc()
            emit_error("Server error starting processing for one or more images.", sid)
            
# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App (Flask + SocketIO + Eventlet) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default Flask SECRET_KEY!")
    # Removed Google API Key check
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Ext: {ALLOWED_EXTENSIONS}")
    print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async: {socketio.async_mode}")
    print(f"   * Roboflow Key: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Google API Key: {'Removed/Unused'}")
    print(f"   * Upload Limit: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f} MB")
    print(f"   * SocketIO Ping Timeout: {socketio.ping_timeout}s")
    print(f"   * Starting server http://0.0.0.0:{port}")
    print("   * NOTE: For production, consider using Gunicorn with eventlet workers or Waitress.")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}")
        sys.exit(1)
