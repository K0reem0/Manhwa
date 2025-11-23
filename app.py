# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
from requests.exceptions import RequestException
import traceback
import uuid
import shutil
import io
import math
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
from werkzeug.utils import secure_filename
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid

eventlet.monkey_patch()

# --- Configuration & Setup ---
load_dotenv()
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Roboflow Endpoints and Settings ---
# NEW Bubble Detection API (1% confidence threshold applied in API call)
ROBOFLOW_BUBBLE_DETECT_URL = 'https://serverless.roboflow.com/manga-speech-bubble-detection-1rbgq/15'
ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1' # Using the original text model

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default_secret')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50 MB

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
    print("✔️ Directories verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR creating directories: {e}")
    sys.exit(1)

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

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60, confidence=None):
    """Fetches predictions from a Roboflow model."""
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown Model"
    print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    url = f"{endpoint_url}?api_key={api_key}"
    if confidence is not None:
        url += f"&confidence={confidence}" # Append confidence for bubble detection

    try:
        response = requests.post(url, data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status(); data = response.json(); predictions = data.get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err: print(f"❌ Roboflow ({model_name}) Timeout: {timeout_err}"); raise ConnectionError(f"Roboflow API ({model_name}) timed out.") from timeout_err
    except requests.exceptions.HTTPError as http_err: print(f"❌ Roboflow ({model_name}) HTTP Error: Status {http_err.response.status_code}"); print(f"   Response: {http_err.response.text[:200]}"); raise ConnectionError(f"Roboflow API ({model_name}) failed (Status {http_err.response.status_code}).") from http_err
    except requests.exceptions.RequestException as req_err: print(f"❌ Roboflow ({model_name}) Request Error: {req_err}"); raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e: print(f"❌ Roboflow ({model_name}) Unexpected Error: {e}"); traceback.print_exc(limit=2); raise RuntimeError(f"Unexpected error during Roboflow ({model_name}).") from e

def is_point_in_polygon(x, y, poly_coords):
    """Checks if a point (x, y) is inside a polygon."""
    poly = Polygon(poly_coords)
    return poly.contains(Polygon([(x, y)]))

def merge_predictions(predictions_list):
    """Merges all text prediction polygons into a single mask."""
    if not predictions_list: return None, 0

    all_points = []
    for pred in predictions_list:
        points = pred.get("points", [])
        if len(points) >= 3:
            all_points.append([(int(p["x"]), int(p["y"])) for p in points])

    if not all_points: return None, 0
    return all_points, len(all_points)

# --- NEW: Image Splitting and Merging Logic ---

MAX_DETECTION_HEIGHT = 10000
MAX_INPAINT_HEIGHT = 7000
MIN_INPAINT_HEIGHT = 5000
LARGE_IMAGE_THRESHOLD = 15000

def get_bubble_predictions_for_large_image(image, sid):
    """Splits large image for bubble detection and aggregates predictions."""
    h_img, w_img = image.shape[:2]
    if h_img <= LARGE_IMAGE_THRESHOLD:
        print("   Image is small enough, processing without splitting.")
        retval, buffer = cv2.imencode('.jpg', image); b64_image = base64.b64encode(buffer).decode('utf-8')
        # Confidence=0.01 (1%) for the specific bubble detection API
        return get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_image, confidence=0.01)

    print(f"   Image height {h_img} > {LARGE_IMAGE_THRESHOLD}. Splitting for detection...")
    segment_predictions = []
    
    # Use 10000px sections for detection
    num_segments = math.ceil(h_img / MAX_DETECTION_HEIGHT)
    
    for i in range(num_segments):
        start_y = i * MAX_DETECTION_HEIGHT
        end_y = min((i + 1) * MAX_DETECTION_HEIGHT, h_img)
        segment = image[start_y:end_y, :]
        
        if segment.size == 0: continue
        
        emit_progress(1, f"Detecting bubbles in segment {i+1}/{num_segments}...", 10 + int(30 * (i+1) / num_segments), sid)
        
        try:
            retval, buffer = cv2.imencode('.jpg', segment); b64_segment = base64.b64encode(buffer).decode('utf-8')
            preds = get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_segment, confidence=0.01)
            
            # Re-offset the coordinates to the original image coordinates
            for pred in preds:
                for point in pred.get("points", []):
                    point["y"] += start_y
            segment_predictions.extend(preds)
            
        except Exception as e:
            print(f"   ❌ Error detecting bubbles in segment {i+1}: {e}")
            # Continue to next segment
            
    print(f"   Aggregated {len(segment_predictions)} bubbles from all segments.")
    return segment_predictions

def split_image_safely(image, predictions):
    """Splits image into smaller chunks (5000-7000px) ensuring no bubble polygon is cut."""
    h_img, w_img = image.shape[:2]
    if h_img <= MAX_INPAINT_HEIGHT:
        return [{'crop': image, 'offset_y': 0, 'preds': predictions}]

    # Convert predictions to a list of shapely Polygons for efficient boundary checks
    bubble_polygons = []
    for pred in predictions:
        points = pred.get("points", [])
        if len(points) >= 3:
            coords = [(p["x"], p["y"]) for p in points]
            try:
                poly = Polygon(coords)
                if not poly.is_valid: poly = make_valid(poly)
                if isinstance(poly, MultiPolygon): poly = max(poly.geoms, key=lambda p: p.area, default=None)
                if poly and not poly.is_empty:
                    bubble_polygons.append(poly)
            except Exception as e:
                print(f"   ⚠️ Warning: Failed to create/validate polygon: {e}")

    print(f"   Splitting image safely for inpainting based on {len(bubble_polygons)} bubbles.")
    
    split_sections = []
    current_y = 0
    
    while current_y < h_img:
        target_split = current_y + MAX_INPAINT_HEIGHT
        
        # If the remaining height is too small, just take the rest
        if h_img - current_y <= MIN_INPAINT_HEIGHT:
            split_y = h_img
        else:
            # Find the lowest safe split point between MIN_INPAINT_HEIGHT and MAX_INPAINT_HEIGHT
            # Search backward from target_split to current_y + MIN_INPAINT_HEIGHT
            safe_split_found = False
            for split_y in range(target_split, current_y + MIN_INPAINT_HEIGHT -1, -1):
                if split_y >= h_img: continue
                
                is_safe = True
                for poly in bubble_polygons:
                    miny_poly, maxy_poly = int(poly.bounds[1]), int(poly.bounds[3])
                    # Check if the split line cuts through any bubble
                    if miny_poly < split_y < maxy_poly:
                        is_safe = False
                        break
                
                if is_safe:
                    safe_split_found = True
                    break
            
            if not safe_split_found:
                 # Fallback: Find the first safe spot outside of all current bubbles below current_y
                 # This should ideally not happen if MAX_INPAINT_HEIGHT is small enough
                 print(f"   ⚠️ Warning: Could not find safe split between {current_y+MIN_INPAINT_HEIGHT} and {target_split}. Finding first safe spot...")
                 
                 # Look ahead up to 1500px past max allowed split just in case
                 for split_y in range(target_split, min(h_img, target_split + 1500)): 
                     is_safe = True
                     for poly in bubble_polygons:
                         miny_poly, maxy_poly = int(poly.bounds[1]), int(poly.bounds[3])
                         if miny_poly < split_y < maxy_poly:
                             is_safe = False
                             break
                     if is_safe:
                         safe_split_found = True
                         break
                 
                 if not safe_split_found:
                      # If still unsafe, take the max crop and hope for the best (will likely cut a bubble)
                      split_y = target_split
                      print("   ❌ CRITICAL: Failed to find safe split. Cutting at max height.")
                 elif split_y > h_img: split_y = h_img

        split_y = min(split_y, h_img)
        
        # Extract the crop
        crop = image[current_y:split_y, :]
        
        # Filter and offset predictions for the current crop
        crop_preds = []
        for pred in predictions:
            points = pred.get("points", [])
            # Simple check: Does the bubble's bounding box overlap the crop?
            coords_y = [p["y"] for p in points]
            miny, maxy = min(coords_y), max(coords_y)
            
            if miny < split_y and maxy > current_y:
                # Need to check if a significant part of the polygon is in the crop
                # For simplicity and speed, we will rely on the split line being safe.
                # Offset coordinates for the current crop
                new_points = []
                for p in points:
                    new_points.append({"x": p["x"], "y": p["y"] - current_y})

                new_pred = pred.copy()
                new_pred['points'] = new_points
                crop_preds.append(new_pred)
                
        split_sections.append({
            'crop': crop,
            'offset_y': current_y,
            'preds': crop_preds
        })
        
        current_y = split_y
        
    print(f"   Split image into {len(split_sections)} safe sections for inpainting.")
    return split_sections

def merge_processed_sections(sections):
    """Merges processed image chunks back into a single image."""
    if not sections: return None
    
    # Determine total height
    total_height = sum(section['crop'].shape[0] for section in sections)
    width = sections[0]['crop'].shape[1]
    
    merged_image = np.zeros((total_height, width, 3), dtype=np.uint8) # Assuming BGR
    
    current_y = 0
    for section in sections:
        h, w = section['crop'].shape[:2]
        merged_image[current_y:current_y + h, :] = section['crop']
        current_y += h
        
    return merged_image

# --- Main Processing Task (Modified) ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """
    Main background task to process a single image (Whitewashing only).
    """
    print(f"ℹ️ SID {sid}: Starting image processing task for {os.path.basename(image_path)}")
    start_time = time.time(); final_image_np = None; result_data = {}; image = None
    final_output_path = ""

    try:
        emit_progress(0, "Loading image...", 5, sid);
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path} (Task Level).")
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        
        emit_progress(1, "Detecting bubbles (and splitting if large)...", 10, sid);
        # Step 1: Detect Bubbles (with large image splitting)
        all_bubble_predictions = get_bubble_predictions_for_large_image(image.copy(), sid)
        
        if not all_bubble_predictions:
            print(f"   No speech bubbles detected.")
            emit_progress(4, "No bubbles detected, skipping text removal.", 95, sid); 
            final_image_np = image.copy(); 
            output_filename = f"{output_filename_base}_no_changes.jpg"; 
            final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); 
            result_data = {'mode': 'cleaned', 'imageUrl': f'/results/{output_filename}'}

        else:
            # Step 2: Split image safely based on bubble locations
            image_sections = split_image_safely(image.copy(), all_bubble_predictions)
            processed_sections = []
            
            total_sections = len(image_sections)
            base_progress = 40; max_progress = 90

            # Step 3: Process each section
            for i, section in enumerate(image_sections):
                current_section_progress = base_progress + int(((i + 1) / total_sections) * (max_progress - base_progress))
                emit_progress(2, f"Processing section {i + 1}/{total_sections}...", current_section_progress, sid)
                
                section_image = section['crop']
                section_h, section_w = section_image.shape[:2]
                section_predictions = section['preds']
                
                # a) Get all text predictions within this section
                retval, buffer_text = cv2.imencode('.jpg', section_image)
                if not retval or buffer_text is None: raise ValueError("Failed encode text detect segment.")
                b64_image_text = base64.b64encode(buffer_text).decode('utf-8')
                
                text_predictions = []
                try:
                    text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image_text)
                    print(f"   Found {len(text_predictions)} text areas in section {i+1}.")
                except Exception as rf_err:
                     print(f"❌ SID {sid}: Error during Roboflow text detection in section {i+1}: {rf_err}. Skipping text removal for this section.")

                # b) Create the text mask (for inpainting)
                text_mask = np.zeros((section_h, section_w), dtype=np.uint8)
                polygons_drawn = 0
                
                # Text polygons
                for pred in text_predictions:
                     points = pred.get("points", []);
                     if len(points) >= 3:
                         polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                         polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, section_w - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, section_h - 1)
                         try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                         except Exception as fill_err: print(f"⚠️ Warn: Error drawing text polygon: {fill_err}")

                # c) Perform Inpainting
                if np.any(text_mask):
                     print(f"   Inpainting detected text areas in section {i+1}...")
                     inpainted_section_cv = cv2.inpaint(section_image, text_mask, 10, cv2.INPAINT_NS)
                     if inpainted_section_cv is None: raise RuntimeError(f"cv2.inpaint returned None for section {i+1}")
                     print(f"   Inpainting complete for section {i+1}.")
                else: 
                    inpainted_section_cv = section_image.copy()
                    print(f"   No text detected in section {i+1}, skipping inpainting.")
                    
                processed_sections.append({'crop': inpainted_section_cv, 'offset_y': section['offset_y']})
            
            # Step 4: Merge sections back
            emit_progress(3, "Merging processed sections...", 92, sid)
            final_image_np = merge_processed_sections(processed_sections)
            if final_image_np is None: raise RuntimeError("Failed to merge processed sections.")
            
            output_filename = f"{output_filename_base}_cleaned.jpg"; 
            final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename);
            result_data = {'mode': 'cleaned', 'imageUrl': f'/results/{output_filename}'}

        # Final Save
        if final_image_np is not None and final_output_path:
            emit_progress(5, "Saving final image...", 98, sid); save_success = False
            # Save as JPEG with 95% quality
            try:
                 encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                 retval, buffer = cv2.imencode('.jpg', final_image_np, encode_param)
                 if not retval or buffer is None: raise IOError("cv2.imencode failed for JPEG 95")
                 
                 with open(final_output_path, 'wb') as f:
                     f.write(buffer.tobytes())
                 
                 save_success = True
                 print(f"✔️ Saved (JPEG 95%): {final_output_path}")
                 
            except Exception as save_err: 
                 print(f"❌ Final save failed: {save_err}"); emit_error("Failed save final image.", sid)
            
            if save_success:
                processing_time = time.time() - start_time;
                print(f"✔️ SID {sid} Complete {processing_time:.2f}s for {os.path.basename(image_path)}.");
                emit_progress(6, f"Complete ({processing_time:.2f}s).", 100, sid)
                result_data['original_filename'] = os.path.basename(image_path)
                result_data['is_zip_batch'] = False
                socketio.emit('processing_complete', result_data, room=sid)
            else:
                print(f"❌❌❌ SID {sid}: Critical Error: Could not save image {final_output_path}")
        elif not final_output_path:
            print(f"❌ SID {sid}: Aborted before output path set.")
        else:
            print(f"❌ SID {sid}: No final image data."); emit_error("Internal error: No final image.", sid)

    except Exception as e:
        print(f"❌❌❌ SID {sid}: UNHANDLED FATAL ERROR in task: {e}")
        tracebox.print_exc()
        emit_error(f"Unexpected server error during processing ({type(e).__name__}).", sid)
    finally:
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 SID {sid}: Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ SID {sid}: Error cleaning up {image_path}: {cleanup_err}")

# --- Flask Routes (Unchanged) ---
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

# --- Upload Routes (Kept for compatibility, now expecting .jpg/png/etc only) ---
@app.route('/upload', methods=['POST'])
def handle_upload():
    # ... (Keep the original handle_upload for single files, simplified logic) ...
    temp_log_id = f"upload_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST upload request ---")
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS: return jsonify({'error': f'Invalid file type: {ext}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    else: return jsonify({'error': 'File has no extension'}), 400
    unique_id = uuid.uuid4(); input_filename = f"{unique_id}.{ext}"; output_filename_base = f"{unique_id}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True); file.save(upload_path)
        file_size_kb = os.path.getsize(upload_path) / 1024
        print(f"[{temp_log_id}] ✔️ File saved via POST: {upload_path} ({file_size_kb:.1f} KB)")
        return jsonify({'message': 'File uploaded successfully', 'output_filename_base': output_filename_base, 'saved_filename': input_filename}), 200
    except IOError as io_err:
        print(f"[{temp_log_id}] ❌ Upload Error: File write error: {io_err}")
        if os.path.exists(upload_path):
            try: os.remove(upload_path); print(f"[{temp_log_id}] 🧹 Cleaned up partial file after IO error.")
            except Exception: pass
        return jsonify({'error': 'Server error saving file'}), 500
    except Exception as e:
        print(f"[{temp_log_id}] ❌ Upload Error: Unexpected error during save: {e}"); traceback.print_exc()
        if os.path.exists(upload_path):
             try: os.remove(upload_path); print(f"[{temp_log_id}] 🧹 Cleaned up file after unexpected save error.")
             except Exception: pass
        return jsonify({'error': 'Unexpected server error during upload'}), 500

@app.route('/upload_zip', methods=['POST'])
def handle_zip_upload():
    # ... (Keep the original handle_zip_upload for batch files) ...
    temp_log_id = f"upload_zip_{uuid.uuid4()}"
    print(f"--- [{temp_log_id}] Received POST zip upload request ---")
    if 'file' not in request.files: return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if not file.filename.lower().endswith('.zip'): return jsonify({'error': 'Invalid file type. Please upload a .zip file.'}), 400

    temp_dir_for_extraction = None
    try:
        zip_file_bytes = file.read(); zip_file = zipfile.ZipFile(io.BytesIO(zip_file_bytes)); extracted_images_info = []
        temp_dir_for_extraction = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_zip_{uuid.uuid4()}"); os.makedirs(temp_dir_for_extraction, exist_ok=True)
        print(f"[{temp_log_id}] Created temporary directory for extraction: {temp_dir_for_extraction}")
        for file_info in zip_file.infolist():
            if file_info.is_dir(): continue
            filename = os.path.basename(file_info.filename); ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            if ext in ALLOWED_EXTENSIONS:
                extracted_file_path = os.path.join(temp_dir_for_extraction, secure_filename(filename))
                try:
                    with zip_file.open(file_info) as source, open(extracted_file_path, "wb") as target: target.write(source.read())
                    print(f"[{temp_log_id}] Extracted image: {extracted_file_path}")
                    extracted_images_info.append({'original_filename': filename, 'saved_path': extracted_file_path, 'output_base': f"zip_img_{uuid.uuid4()}"})
                except Exception as extract_err: print(f"[{temp_log_id}] ⚠️ Warning: Could not extract {filename}: {extract_err}")
            else: print(f"[{temp_log_id}] Skipping non-image file in zip: {filename}")
        if not extracted_images_info:
            print(f"[{temp_log_id}] ❌ Upload Error: No valid images found in the zip file.")
            if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
            return jsonify({'error': 'No valid images found in the zip file.'}), 400
        print(f"[{temp_log_id}] Successfully extracted {len(extracted_images_info)} images.")
        return jsonify({'message': 'Zip file uploaded and images extracted successfully', 'images_to_process': extracted_images_info}), 200
    except zipfile.BadZipFile:
        print(f"[{temp_log_id}] ❌ Upload Error: Bad zip file.")
        if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
        return jsonify({'error': 'The uploaded file is not a valid zip archive.'}), 400
    except Exception as e:
        print(f"[{temp_log_id}] ❌ Upload Error: Unexpected error during zip processing: {e}"); traceback.print_exc()
        if temp_dir_for_extraction and os.path.exists(temp_dir_for_extraction): shutil.rmtree(temp_dir_for_extraction)
        return jsonify({'error': 'Unexpected server error during zip upload processing.'}), 500


# --- SocketIO Event Handlers (Mode set to 'cleaned') ---
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    """Handles the start of a single-image processing task (Whitewash mode only)."""
    sid = request.sid
    print(f"\n--- Received 'start_processing' event SID: {sid} ---")
    if not isinstance(data, dict): emit_error("Invalid request data.", sid); return
    output_filename_base = data.get('output_filename_base')
    saved_filename = data.get('saved_filename')
    # Force mode to 'cleaned' as per user request
    mode = 'cleaned'
    print(f"   Mode: '{mode}' (Forced)")
    if not output_filename_base or not saved_filename:
        emit_error('Missing or invalid parameters.', sid); return
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(saved_filename))
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        print(f"❌ ERROR SID {sid}: File not found at path: {upload_path}")
        emit_error(f"Uploaded file '{saved_filename}' not found on server.", sid); return
    
    try:
        socketio.start_background_task(process_image_task, image_path=upload_path, output_filename_base=output_filename_base, mode=mode, sid=sid)
        print(f"   ✔️ Task initiated SID {sid} for {saved_filename}.")
        socketio.emit('processing_started', {'message': 'File received, processing started...'}, room=sid)
    except Exception as task_start_err:
        print(f"❌ CRITICAL SID {sid}: Failed to start background task: {task_start_err}"); traceback.print_exc()
        emit_error(f"Server error starting image processing task.", sid)

@socketio.on('start_batch_processing')
def handle_start_batch_processing(data):
    """Handles the start of a batch-image processing task (Whitewash mode only)."""
    sid = request.sid
    print(f"\n--- Received 'start_batch_processing' event SID: {sid} ---")
    if not isinstance(data, dict): emit_error("Invalid request data.", sid); return
    images_to_process = data.get('images_to_process', [])
    # Force mode to 'cleaned' as per user request
    mode = 'cleaned'
    if not isinstance(images_to_process, list) or not images_to_process:
        emit_error('Invalid or empty image list for batch processing.', sid); return

    print(f"   Initiating batch processing for {len(images_to_process)} images. Mode: '{mode}' (Forced)")
    socketio.emit('batch_started', {'total_images': len(images_to_process)}, room=sid)

    for img_info in images_to_process:
        try:
            image_path = img_info.get('saved_path'); output_base = img_info.get('output_base')
            original_filename = img_info.get('original_filename', 'unknown')
            if not image_path or not output_base:
                print(f"   ⚠️ Skipping image {original_filename}: missing path or output base."); emit_error(f"Skipping image '{original_filename}' due to missing data.", sid); continue
            if not os.path.exists(image_path) or not os.path.isfile(image_path):
                print(f"❌ ERROR SID {sid}: Image file not found: {image_path}"); emit_error(f"Image '{original_filename}' not found on server. Skipping.", sid); continue

            # Start a separate background task for each image
            socketio.start_background_task(process_image_task, image_path=image_path, output_filename_base=output_base, mode=mode, sid=sid)
            print(f"   ✔️ Task initiated for image '{original_filename}'.")
        except Exception as e:
            print(f"❌ CRITICAL SID {sid}: Failed to start task for an image: {e}"); traceback.print_exc()
            emit_error("Server error starting processing for one or more images.", sid)
            
# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    print("--- Starting Manga Whitewashing Web App (Flask + SocketIO + Eventlet) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! ███")
    print("   * Note: Translation and text drawing functionality has been removed.")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Starting server http://0.0.0.0:{port}")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}"); sys.exit(1)
