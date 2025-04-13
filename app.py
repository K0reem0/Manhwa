# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import requests
import base64
import time
from PIL import Image, ImageDraw, ImageFont # Keep ImageFont for type hinting if needed
from requests.exceptions import RequestException
import re
from shapely.geometry import Polygon
from shapely.validation import make_valid
import uuid # To generate unique filenames
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet # Required for SocketIO async mode
import json # For sending complex data like the table
import traceback # For detailed error logging

eventlet.monkey_patch() # Patch standard libraries for async operations

# --- IMPORT YOUR MODULE ---
try:
    import text_formatter # Assuming text_formatter.py is in the same directory
    print("✔️ Successfully imported 'text_formatter.py'.")
except ImportError:
    print("❌ ERROR: Cannot import 'text_formatter.py'. Make sure the file exists.")
    # Define dummy class and functions *only* if the import fails.
    # Ensure this class definition is correctly indented under the 'except' block.
    class DummyTextFormatter:
        """A fallback class if text_formatter module fails to import."""
        def __init__(self):
            print("⚠️ WARNING: Initializing DummyTextFormatter. Real text formatting will NOT work.")
            self._font_path = None # Keep track internally

        def set_arabic_font_path(self, path):
            print(f"   (Dummy) Ignoring font path: {path}")
            self._font_path = path # Store it, though we won't use it effectively

        def get_font(self, size):
            print(f"   (Dummy) Attempting to get font size {size}.")
            # Try loading a basic default font if Pillow is available
            try:
                from PIL import ImageFont
                try:
                    # Try loading the specific font if path was set, even if formatting fails
                    if self._font_path and os.path.exists(self._font_path):
                         return ImageFont.truetype(self._font_path, size)
                    # Fallback to default font included with Pillow (may not exist everywhere)
                    return ImageFont.load_default()
                except IOError:
                    print("   (Dummy) Could not load default PIL font or specified font.")
                    return None # Return None if font loading fails
            except ImportError:
                 print("   (Dummy) PIL.ImageFont not available.")
                 return None # Return None if Pillow isn't installed

        def format_arabic_text(self, text):
            print("   (Dummy) Returning raw text (no Arabic reshaping).")
            return text # Return raw text, no processing

        def layout_balanced_text(self, draw, text, font, target_width):
             print("   (Dummy) Returning raw text for layout (no wrapping).")
             # Basic fallback: just return the un-wrapped text
             return text

    # This line MUST also be indented under the 'except' block, after the class definition.
    text_formatter = DummyTextFormatter()
    print("⚠️ WARNING: Using dummy 'text_formatter' due to import error. Text rendering might fail or look incorrect.")


load_dotenv() # Load environment variables from .env

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Set a strong secret key in your .env file or environment
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key_CHANGE_ME')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit upload size
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=False, engineio_logger=False) # Disable default SocketIO logs for cleaner output, enable if needed

# --- Ensure directories exist ---
# Do this once at startup
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Uploads directory '{UPLOAD_FOLDER}' verified/created.")
    print(f"✔️ Results directory '{RESULT_FOLDER}' verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR: Could not create directories '{UPLOAD_FOLDER}' or '{RESULT_FOLDER}'. Check permissions. Error: {e}")
    # Optionally exit if directories are essential and cannot be created
    # sys.exit(1)

# --- Font Setup ---
def setup_font():
    """Finds the font file path and sets it using the text_formatter object."""
    font_path_to_set = None
    try:
        # --- Find Font Path ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Look inside a 'fonts' subdirectory relative to the script
        potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf") # Adjust font name if different
        if os.path.exists(potential_path):
            font_path_to_set = potential_path
        # Fallback: Look in 'fonts' subdirectory relative to current working directory
        elif os.path.exists(os.path.join(".", "fonts", "66Hayah.otf")):
             font_path_to_set = os.path.join(".", "fonts", "66Hayah.otf")

        # --- Set Font Path using text_formatter (real or dummy) ---
        if font_path_to_set:
            print(f"ℹ️ Font found: '{font_path_to_set}'. Setting path via text_formatter.")
            # Call the method directly - it works on both real and dummy objects
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found. Using default font setting via text_formatter.")
            # Tell the formatter (real or dummy) to use its default/None path
            text_formatter.set_arabic_font_path(None)

    except Exception as e:
        # Catch errors specifically related to finding the path (e.g., os functions)
        print(f"❌ Error during font *path finding*: {e}. Using default font setting.")
        # Ensure we attempt to set the path to None in the formatter on error
        try:
            text_formatter.set_arabic_font_path(None)
        except Exception as E2:
             print(f"❌ Error setting font path to None after another error: {E2}")

setup_font()

# --- Constants ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا". حافظ على المعنى والنبرة الأصلية.'

# --- Helper Functions ---
# (emit_progress, emit_error, get_roboflow_predictions, extract_translation, ask_luminai)
# (find_optimal_text_settings_final, draw_text_on_layer)
# --- Ensure these functions are defined correctly as in previous versions ---
# --- Add robust error handling within them if possible ---

def emit_progress(step, message, percentage, sid):
    """Helper to emit progress updates to a specific client."""
    # print(f"SID: {sid} | Progress ({step}): {message} ({percentage}%)") # Reduced logging noise
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)

def emit_error(message, sid):
    """Helper to emit error messages to a specific client."""
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

# --- Placeholder definitions for core logic functions (REPLACE WITH YOUR ACTUAL WORKING CODE) ---
# --- It's critical these functions handle errors gracefully (e.g., return None or raise specific exceptions) ---
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=35):
    print(f"ℹ️ Calling Roboflow: {endpoint_url[:50]}...") # Log API call start
    # REPLACE with your actual working function, including error handling
    try:
        # ... (your requests.post call) ...
        # response.raise_for_status()
        # data = response.json()
        # print(f"✔️ Roboflow response received. Predictions: {len(data.get('predictions', []))}")
        # return data.get("predictions", [])
        print("⚠️ Using placeholder Roboflow data.")
        return [] # Placeholder
    except Exception as e:
        print(f"❌ Roboflow call failed: {e}")
        raise ConnectionError(f"Roboflow API request failed.") from e # Raise specific error

def extract_translation(text):
     if not isinstance(text, str): return ""
     match = re.search(r'"(.*?)"', text, re.DOTALL)
     if match: return match.group(1).strip()
     return text.strip('"').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    print("ℹ️ Calling LuminAI...")
    # REPLACE with your actual working function, including error handling
    try:
        # ... (your requests.post loop) ...
        # print("✔️ LuminAI response received.")
        # return translation
        print("⚠️ Using placeholder LuminAI translation.")
        return "ترجمة تجريبية" # Placeholder
    except Exception as e:
        print(f"❌ LuminAI call failed: {e}")
        # Don't raise here, just return empty string or default
        return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    print("ℹ️ Finding optimal text settings...")
     # REPLACE with your actual working function. Critical to handle font loading errors.
    try:
        # font = text_formatter.get_font(size) # Inside your loop
        # if font is None: continue # Skip if font failed to load for this size
        # ... (rest of your logic) ...
        # print("✔️ Optimal settings found.")
        # return settings_dict
        print("⚠️ Using placeholder text settings.")
        font = text_formatter.get_font(20) if text_formatter else None
        if font and initial_shrunk_polygon.is_valid and not initial_shrunk_polygon.is_empty:
            minx, miny, _, _ = initial_shrunk_polygon.bounds
            return {'text': text, 'font': font, 'x': int(minx) + 5, 'y': int(miny) + 5, 'font_size': 20}
        return None # Indicate failure
    except Exception as e:
        print(f"❌ Error in find_optimal_text_settings_final: {e}")
        return None

def draw_text_on_layer(text_settings, image_size):
    print("ℹ️ Drawing text layer...")
     # REPLACE with your actual working function
    try:
        # ... (Your PIL drawing logic using text_settings) ...
        # print("✔️ Text layer drawn.")
        # return text_layer
        print("⚠️ Using placeholder text drawing.")
        text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
        # Optional: draw a simple placeholder if needed for testing
        # draw_on_layer = ImageDraw.Draw(text_layer)
        # draw_on_layer.text((10, 10), "Placeholder", fill=(255,0,0,255))
        return text_layer
    except Exception as e:
        print(f"❌ Error in draw_text_on_layer: {e}")
        # Return an empty layer on error
        return Image.new('RGBA', image_size, (0, 0, 0, 0))
# --- End Placeholders ---

# --- Main Processing Task ---
# (Keep the process_image_task function as defined in the previous response)
# --- Ensure it uses the actual core logic functions above, not the placeholders if possible ---
# --- Add error handling within the task itself ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """ Core logic: Cleans, optionally translates, and draws text based on mode. """
    start_time = time.time()
    inpainted_image = None
    final_image_np = None
    translations_list = []
    final_output_path = ""
    result_data = {}

    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image at {image_path}")

        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError("Unsupported image channel format")

        h_img, w_img = image.shape[:2]
        original_image_for_cropping = image.copy()
        result_image = image.copy()

        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        _, buffer = cv2.imencode('.jpg', image)
        if buffer is None: raise ValueError("Failed to encode image for text detection.")
        b64_image = base64.b64encode(buffer).decode('utf-8')

        # Wrap Roboflow call in try-except within the task
        try:
            text_predictions = get_roboflow_predictions(
                'https://serverless.roboflow.com/text-detection-w0hkg/1',
                ROBOFLOW_API_KEY, b64_image
            )
        except Exception as rf_err:
            print(f"⚠️ Roboflow text detection failed: {rf_err}. Proceeding without inpainting.")
            emit_progress(1, f"Text detection failed ({type(rf_err).__name__}), skipping removal.", 15, sid)
            text_predictions = [] # Ensure it's an empty list

        emit_progress(1, f"Found {len(text_predictions)} text areas. Masking...", 15, sid)

        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # ... (mask drawing loop - keep as before) ...
        polygons_drawn = 0
        for pred in text_predictions:
             points = pred.get("points", [])
             if len(points) >= 3:
                 polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                 polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1)
                 polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                 try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                 except Exception as fill_err: print(f"⚠️ Warn: Error drawing text poly: {fill_err}")


        if np.any(text_mask):
            emit_progress(1, "Inpainting text areas...", 20, sid)
            inpainted_image = cv2.inpaint(result_image, text_mask, 5, cv2.INPAINT_TELEA)
            emit_progress(1, "Inpainting complete.", 25, sid)
        else:
            emit_progress(1, "No text found/detected to remove.", 25, sid)
            inpainted_image = result_image.copy()

        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        _, buffer_bubble = cv2.imencode('.jpg', inpainted_image) # Use inpainted
        if buffer_bubble is None: raise ValueError("Failed to encode inpainted image for bubble detection.")
        b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8')

        try:
             bubble_predictions = get_roboflow_predictions(
                  'https://outline.roboflow.com/yolo-0kqkh/2',
                  ROBOFLOW_API_KEY, b64_bubble
             )
        except Exception as rf_err:
             # If bubble detection fails, we can't proceed with translation/drawing
             print(f"❌ Bubble detection failed: {rf_err}. Aborting bubble processing.")
             raise RuntimeError(f"Bubble detection API failed ({type(rf_err).__name__}).") from rf_err

        emit_progress(2, f"Found {len(bubble_predictions)} speech bubbles.", 40, sid)

        # --- Bubble Processing Logic (keep the rest as before) ---
        if not bubble_predictions:
             # ... (handle no bubbles case - save cleaned image) ...
             emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        else:
            # ... (bubble processing loop as in the previous version) ...
            # ... (ensure try-except blocks inside the loop for individual bubble errors) ...
            # ... (calls to ask_luminai, text_formatter, find_optimal_text_settings_final, draw_text_on_layer) ...
            # --- CRITICAL: Ensure these called functions handle their own errors gracefully ---
            image_pil = None # Initialize PIL image reference for 'auto' mode
            if mode == 'auto':
                try: image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                except Exception as pil_conv_err: raise RuntimeError(f"PIL conversion failed: {pil_conv_err}") from pil_conv_err
                image_size = image_pil.size
                temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', image_size))

            bubble_count = len(bubble_predictions)
            processed_count = 0
            base_progress = 45
            max_progress_bubbles = 90

            for i, pred in enumerate(bubble_predictions):
                # ... (progress calculation) ...
                current_bubble_progress = base_progress + int((i / bubble_count) * (max_progress_bubbles - base_progress))
                emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                # --- Individual Bubble Try-Except ---
                try:
                    points = pred.get("points", [])
                    if len(points) < 3: continue
                    coords = [(int(p["x"]), int(p["y"])) for p in points]

                    bubble_polygon = Polygon(coords)
                    # ... (polygon validation) ...
                    if not bubble_polygon.is_valid: bubble_polygon = make_valid(bubble_polygon)
                    if bubble_polygon.geom_type == 'MultiPolygon': bubble_polygon = max(bubble_polygon.geoms, key=lambda p: p.area, default=None)
                    if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty: continue

                    minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                    # ... (crop original image for context - same logic) ...
                    minx_c = max(0, minx - 5); miny_c = max(0, miny - 5)
                    maxx_c = min(w_img, maxx + 5); maxy_c = min(h_img, maxy + 5)
                    if maxx_c <= minx_c or maxy_c <= miny_c: continue
                    bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                    if bubble_crop.size == 0: continue
                    _, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop)
                    if crop_buffer_enc is None: continue
                    crop_bytes = crop_buffer_enc.tobytes()

                    translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, sid=sid)
                    if not translation: translation = "[ترجمة فارغة]"

                    if mode == 'extract':
                        translations_list.append({'id': i + 1, 'translation': translation})
                        processed_count += 1
                    elif mode == 'auto':
                        if translation == "[ترجمة فارغة]": continue # Skip drawing empty
                        # --- Call text processing functions (ensure they handle errors) ---
                        arabic_text = text_formatter.format_arabic_text(translation) if text_formatter else translation
                        if not arabic_text: continue
                        # ... (polygon shrinking logic - same) ...
                        initial_buffer = max(3.0, (maxx - minx + maxy - miny) / 2 * 0.08)
                        text_poly = bubble_polygon # Default
                        try:
                           shrunk = bubble_polygon.buffer(-initial_buffer, join_style=2)
                           if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                           else:
                               shrunk = bubble_polygon.buffer(-2.0, join_style=2)
                               if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                        except Exception: pass # Ignore buffer errors, use default

                        text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                        if text_settings:
                            text_layer = draw_text_on_layer(text_settings, image_size)
                            if text_layer: # Check if drawing succeeded
                                 image_pil.paste(text_layer, (0, 0), text_layer)
                                 processed_count += 1
                            else: print(f"⚠️ Text layer drawing failed for bubble {i+1}")
                        else: print(f"⚠️ Could not fit text for bubble {i+1}")

                except Exception as bubble_err:
                     # Log error for specific bubble but continue loop
                     print(f"❌ Error processing bubble {i + 1}: {bubble_err}\n---")
                     traceback.print_exc(limit=1) # Print short traceback
                     print("---")
                     emit_progress(3, f"Skipping bubble {i+1} due to error.", current_bubble_progress + 1, sid) # Update progress slightly


            # === Step 4: Finalize based on mode ===
            # ... (same logic as before to set final_image_np and result_data) ...
            if mode == 'extract':
                 emit_progress(4, f"Finished extracting text for {processed_count}/{bubble_count} bubbles.", 95, sid)
                 final_image_np = inpainted_image
                 output_filename = f"{output_filename_base}_cleaned.jpg"
                 final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                 result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
            elif mode == 'auto':
                 emit_progress(4, f"Finished drawing text for {processed_count}/{bubble_count} bubbles.", 95, sid)
                 if image_pil: # Ensure PIL image exists
                      final_image_rgb = image_pil.convert('RGB')
                      final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR)
                 else: # Fallback if PIL conversion failed earlier
                      final_image_np = inpainted_image # Use cleaned image if drawing failed
                 output_filename = f"{output_filename_base}_translated.jpg"
                 final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                 result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}


        # === Step 5: Save Final Image ===
        emit_progress(5, "Saving final image...", 98, sid)
        # ... (same saving logic with PIL fallback) ...
        if final_image_np is None: raise RuntimeError("Final image data is missing.")
        if not final_output_path: raise RuntimeError("Final output path is not set.")
        save_success = cv2.imwrite(final_output_path, final_image_np)
        if not save_success:
             try:
                 print("⚠️ OpenCV save failed, trying PIL fallback...")
                 pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                 pil_img_to_save.save(final_output_path)
                 save_success = True
             except Exception as pil_save_err:
                 raise IOError(f"Failed to save final image using OpenCV and PIL: {pil_save_err}") from pil_save_err

        # === Step 6: Signal Completion ===
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Mode: {mode}. Output: {final_output_path}") # Server log success
        emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
        socketio.emit('processing_complete', result_data, room=sid)

    except Exception as e:
        # Catch all errors within the task and report them
        print(f"❌❌❌ Unhandled error in process_image_task for SID {sid}: {e}")
        traceback.print_exc() # Print full traceback to server console
        emit_error(f"An unexpected server error occurred: {type(e).__name__}. Check server logs.", sid)
    finally:
        # Clean up the originally uploaded file in all cases
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Cleaned up input file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ Error cleaning up file {image_path}: {cleanup_err}")


# --- Flask Routes ---
# (Keep / and /results/<filename> routes as before)
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    """Serves the processed image safely."""
    # Basic check to prevent directory traversal
    if '..' in filename or filename.startswith('/'):
         return "Invalid filename", 400
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=False)

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    """Handles the image upload and starts the background task."""
    sid = request.sid
    # Use print statements for debugging this critical handler
    print(f"\n--- Received 'start_processing' event from SID: {sid} ---") # Mark event start clearly

    # 1. --- Basic Data Validation ---
    if not isinstance(data, dict):
        print(f"   ❗ ERROR: Invalid data type received from {sid}. Expected dict.")
        emit_error("Invalid request data format.", sid)
        return
    print(f"   Data keys received: {list(data.keys())}")
    if 'file' not in data or not isinstance(data['file'], str) or not data['file'].startswith('data:image'):
        print(f"   ❗ ERROR: Missing or invalid 'file' data from {sid}.")
        emit_error('Invalid or missing file data.', sid)
        return
    if 'mode' not in data or data['mode'] not in ['extract', 'auto']:
        print(f"   ❗ ERROR: Missing or invalid 'mode' data from {sid}.")
        emit_error('Invalid or missing processing mode.', sid)
        return
    mode = data['mode']
    print(f"   Mode validated: '{mode}'")

    # 2. --- Directory and Permissions Check ---
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        print(f"   ❗ SERVER ERROR: Upload directory '{upload_dir}' does not exist!")
        emit_error("Server configuration error (upload dir missing).", sid)
        return
    if not os.access(upload_dir, os.W_OK):
        print(f"   ❗ SERVER ERROR: No write permissions for upload directory '{upload_dir}'!")
        emit_error("Server configuration error (upload dir permissions).", sid)
        return
    print(f"   Upload directory '{upload_dir}' exists and is writable.")

    # 3. --- File Decode and Save ---
    upload_path = None # Initialize path variable
    try:
        print(f"   Decoding Base64 data...")
        file_data_str = data['file']
        try:
            header, encoded = file_data_str.split(',', 1)
            file_extension = header.split('/')[1].split(';')[0].split('+')[0] # Handle image/svg+xml etc.
        except ValueError:
            print(f"   ❗ ERROR: Invalid base64 header format from {sid}.")
            emit_error('Invalid image data header.', sid)
            return

        if file_extension not in ALLOWED_EXTENSIONS:
             print(f"   ❗ ERROR: Invalid file extension '{file_extension}' from {sid}.")
             emit_error(f'Invalid file type: {file_extension}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', sid)
             return

        print(f"   File extension: '{file_extension}'")
        file_bytes = base64.b64decode(encoded) # This can raise binascii.Error
        print(f"   Base64 decoded successfully. Size: {len(file_bytes) / 1024:.1f} KB")

        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename_base = f"{unique_id}"
        upload_path = os.path.join(upload_dir, input_filename)

        print(f"   Attempting to write file to: {upload_path}")
        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        print(f"   ✔️ File successfully saved.")

    except (base64.binascii.Error, ValueError) as decode_err:
        print(f"   ❗ ERROR: Base64 decoding failed for {sid}: {decode_err}")
        emit_error(f"Failed to decode image data: {decode_err}", sid)
        return # Stop processing
    except OSError as write_err:
         print(f"   ❗ ERROR: Failed to write file '{upload_path}' for {sid}: {write_err}")
         traceback.print_exc()
         emit_error(f"Server error saving file: {write_err}", sid)
         return # Stop processing
    except Exception as e:
        print(f"   ❗ UNEXPECTED ERROR during file handling for {sid}: {e}")
        traceback.print_exc()
        emit_error(f'Unexpected server error during file upload: {type(e).__name__}', sid)
        return # Stop processing

    # 4. --- Start Background Task ---
    # Only proceed if upload_path is set (meaning file save was successful)
    if upload_path:
        print(f"   Attempting to start background task (Mode: '{mode}') for: {upload_path}")
        try:
            socketio.start_background_task(
                process_image_task,
                upload_path,
                output_filename_base,
                mode,
                sid
            )
            print(f"   ✔️ Background task initiated for SID: {sid}")
            # Send confirmation *after* successfully starting the task
            socketio.emit('processing_started', {'message': 'Upload successful! Processing started...'}, room=sid)
        except Exception as task_err:
            print(f"   ❗ CRITICAL ERROR: Failed to start background task for {sid}: {task_err}")
            traceback.print_exc()
            emit_error(f"Server error initiating processing task: {task_err}", sid)
            # Clean up the uploaded file if task failed to start
            if os.path.exists(upload_path):
                try: os.remove(upload_path); print(f"   🧹 Cleaned up file due to task start failure: {upload_path}")
                except Exception: print(f"   ⚠️ Could not clean up file after task start failure: {upload_path}")
    else:
         # This case should ideally not be reached if error handling above is correct
         print(f"   ❗ LOGIC ERROR: upload_path not set, cannot start background task for {sid}")
         emit_error("Internal server error (upload path missing).", sid)

    print(f"--- Finished handling 'start_processing' event for SID: {sid} ---") # Mark event end clearly


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Flask-SocketIO Application ---")
    # Check essential configurations at startup
    if not ROBOFLOW_API_KEY:
        print("⚠️ WARNING: ROBOFLOW_API_KEY environment variable is not set!")
    if app.config['SECRET_KEY'] == 'a_default_fallback_secret_key_CHANGE_ME':
        print("⚠️ WARNING: Using default FLASK_SECRET_KEY. Set a strong secret key for production!")

    port = int(os.environ.get('PORT', 9000))
    print(f"   * Environment: {os.environ.get('FLASK_ENV', 'production')} (Set FLASK_ENV=development for debug mode)")
    print(f"   * Binding to: host 0.0.0.0, port {port}")
    print(f"   * Async mode: eventlet")
    print(f"   * CORS Allowed Origins: *") # Make sure this is okay for your security needs
    print("--- Ready for connections ---")

    # Use eventlet web server. Set debug=False for production.
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    # For production deployment (e.g., Heroku), use:
    # Procfile: web: gunicorn --worker-class eventlet -w 1 app:app
