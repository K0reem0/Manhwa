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
from shapely.geometry import Polygon
from shapely.validation import make_valid
import uuid # To generate unique filenames
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet # Required for SocketIO async mode
import json # For sending complex data like the table

eventlet.monkey_patch() # Patch standard libraries for async operations

# --- IMPORT YOUR MODULE ---
import text_formatter # Assuming text_formatter.py is in the same directory

load_dotenv() # Load environment variables from .env

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# LUMINAI_API_KEY = os.getenv('LUMINAI_API_KEY') # Assuming LuminAI URL doesn't need key

# --- Flask App Setup ---
app = Flask(__name__)
# IMPORTANT: Set a strong secret key in your .env file or environment
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit upload size
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*") # Allow all origins for simplicity, refine for production

# --- Ensure directories exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Font Setup ---
def setup_font():
    font_path_to_set = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Look inside a 'fonts' subdirectory relative to the script
        potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf")
        if os.path.exists(potential_path):
            font_path_to_set = potential_path
        # Fallback: Look in 'fonts' subdirectory relative to current working directory
        elif os.path.exists(os.path.join(".", "fonts", "66Hayah.otf")):
             font_path_to_set = os.path.join(".", "fonts", "66Hayah.otf")

        if font_path_to_set:
            print(f"ℹ️ WebApp found Arabic font: {font_path_to_set}")
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Warning: WebApp could not find 'fonts/66Hayah.otf'. Using default PIL font.")
            text_formatter.set_arabic_font_path(None) # Explicitly use default
    except Exception as e:
        print(f"ℹ️ Error finding font path: {e}. Using default PIL font.")
        text_formatter.set_arabic_font_path(None)

setup_font()

# --- Constants ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
# LuminAI prompt (keep it concise but clear)
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا". حافظ على المعنى والنبرة الأصلية.'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    """Helper to emit progress updates to a specific client."""
    print(f"SID: {sid} | Progress ({step}): {message} ({percentage}%)")
    socketio.emit('progress_update', {
        'step': step,
        'message': message,
        'percentage': percentage
    }, room=sid)
    socketio.sleep(0.01) # Allow eventlet to process

def emit_error(message, sid):
    """Helper to emit error messages to a specific client."""
    print(f"SID: {sid} | ERROR: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=35):
    """Generic function to call a Roboflow inference endpoint."""
    try:
        response = requests.post(
            f"{endpoint_url}?api_key={api_key}",
            data=image_b64,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json().get("predictions", [])
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Roboflow Request Error ({endpoint_url}): {req_err}")
        raise ConnectionError(f"Network error contacting Roboflow ({endpoint_url[:30]}...).") from req_err
    except Exception as e:
        print(f"❌ Roboflow Unexpected Error ({endpoint_url}): {e}")
        raise RuntimeError(f"Unexpected error during Roboflow request.") from e

# (Keep extract_translation, get_polygon_orientation, find_optimal_text_settings_final, draw_text_on_layer - ensure they use the constants above)
# --- Make sure these functions exist and are correct as per previous versions ---
def extract_translation(text):
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip('"').strip() # Fallback

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    # Using sid just for potential error emission if needed later
    url = "https://luminai.my.id/" # Public endpoint
    payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200:
                result_text = response.json().get("result", "")
                return extract_translation(result_text.strip())
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"LuminAI Rate limit hit. Retrying after {retry_after}s...")
                socketio.sleep(retry_after)
            else:
                print(f"LuminAI request failed: {response.status_code} - {response.text}")
                return "" # Don't retry on other specific errors
        except RequestException as e:
            print(f"Network error during LuminAI request (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"Unexpected error during LuminAI request (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2)
    return ""

# --- find_optimal_text_settings_final and draw_text_on_layer assumed to be defined ---
# These are complex and should be copied from your working script version.
# Ensure `find_optimal_text_settings_final` uses `text_formatter.get_font`
# and `draw_text_on_layer` uses the TEXT_COLOR, SHADOW_COLOR etc. constants.
# Placeholder definitions if you need them:
def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
     # --- THIS IS A COMPLEX FUNCTION - Use your working version ---
     # It should find the best font size and return a dict like:
     # {'text': wrapped_text, 'font': font_object, 'x': draw_x, 'y': draw_y, 'font_size': font_size}
     # Remember to handle errors and return None if no fit is found.
     print("⚠️ Using placeholder for find_optimal_text_settings_final")
     font = text_formatter.get_font(20) # Example fixed size
     if not font or not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty:
         return None
     minx, miny, _, _ = initial_shrunk_polygon.bounds
     return {'text': text, 'font': font, 'x': int(minx) + 5, 'y': int(miny) + 5, 'font_size': 20}

def draw_text_on_layer(text_settings, image_size):
    # --- THIS IS A COMPLEX FUNCTION - Use your working version ---
    # It should draw the text+shadow onto a transparent RGBA layer
    print("⚠️ Using placeholder for draw_text_on_layer")
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
    draw_on_layer = ImageDraw.Draw(text_layer)
    font = text_settings['font']
    text_to_draw = text_settings['text']
    x, y = text_settings['x'], text_settings['y']
    shadow_offset = max(1, text_settings['font_size'] // 18)
    shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)
    # Draw shadow
    draw_on_layer.multiline_text((x + shadow_offset, y + shadow_offset), text_to_draw, font=font, fill=shadow_color_with_alpha, align='center', spacing=4)
    # Draw text
    draw_on_layer.multiline_text((x, y), text_to_draw, font=font, fill=TEXT_COLOR + (255,), align='center', spacing=4)
    return text_layer
# --- End Placeholders ---

# --- Main Processing Task ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """
    Core logic: Cleans, optionally translates, and draws text based on mode.
    mode: 'extract' or 'auto'
    """
    start_time = time.time()
    inpainted_image = None
    final_image_np = None
    translations_list = []
    final_output_path = ""
    result_data = {} # Data to send back to client

    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image at {image_path}")

        # Ensure 3 channels (BGR)
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError("Unsupported image channel format")

        h_img, w_img = image.shape[:2]
        original_image_for_cropping = image.copy() # Keep original for cropping later
        result_image = image.copy() # This will be inpainted/drawn on

        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        _, buffer = cv2.imencode('.jpg', image)
        if buffer is None: raise ValueError("Failed to encode image for text detection.")
        b64_image = base64.b64encode(buffer).decode('utf-8')

        text_predictions = get_roboflow_predictions(
            'https://serverless.roboflow.com/text-detection-w0hkg/1',
            ROBOFLOW_API_KEY, b64_image
        )
        emit_progress(1, f"Found {len(text_predictions)} text areas. Masking...", 15, sid)

        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons_drawn = 0
        for pred in text_predictions:
            points = pred.get("points", [])
            if len(points) >= 3:
                polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                # Clip points to image bounds before drawing polygon
                polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1)
                polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                try:
                    cv2.fillPoly(text_mask, [polygon_np], 255)
                    polygons_drawn += 1
                except Exception as fill_err: print(f"⚠️ Warn: Error drawing text poly: {fill_err}")

        if np.any(text_mask):
            emit_progress(1, "Inpainting text areas...", 20, sid)
            # Use INPAINT_TELEA for potentially better results on manga text
            inpainted_image = cv2.inpaint(result_image, text_mask, 5, cv2.INPAINT_TELEA)
            emit_progress(1, "Inpainting complete.", 25, sid)
        else:
            emit_progress(1, "No text found to remove.", 25, sid)
            inpainted_image = result_image.copy() # Use original if no inpainting needed

        # Now, `inpainted_image` holds the cleaned version.

        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        # Use the *inpainted* image for bubble detection? Or original? Let's try inpainted.
        _, buffer_bubble = cv2.imencode('.jpg', inpainted_image)
        if buffer_bubble is None: raise ValueError("Failed to encode inpainted image for bubble detection.")
        b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8')

        bubble_predictions = get_roboflow_predictions(
             'https://outline.roboflow.com/yolo-0kqkh/2',
             ROBOFLOW_API_KEY, b64_bubble
        )
        emit_progress(2, f"Found {len(bubble_predictions)} speech bubbles.", 40, sid)

        if not bubble_predictions:
            emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
            final_image_np = inpainted_image # Result is the cleaned image
            output_filename = f"{output_filename_base}_cleaned.jpg"
            final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
            # Proceed to saving final_image_np outside the main try block
        else:
            # Bubble processing depends on the mode
            image_pil = None # Initialize PIL image reference for 'auto' mode

            if mode == 'auto':
                # Prepare PIL version of the *inpainted* image for drawing
                try:
                    image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                except Exception as pil_conv_err:
                    raise RuntimeError(f"Failed to convert inpainted image for drawing: {pil_conv_err}") from pil_conv_err
                image_size = image_pil.size
                temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', image_size)) # For measurements

            # === Step 3: Process Each Bubble (Translate +/- Draw) ===
            bubble_count = len(bubble_predictions)
            processed_count = 0
            base_progress = 45 # Starting percentage for this step
            max_progress_bubbles = 90 # Max percentage for this step

            for i, pred in enumerate(bubble_predictions):
                current_bubble_progress = base_progress + int((i / bubble_count) * (max_progress_bubbles - base_progress))
                emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)

                points = pred.get("points", [])
                if len(points) < 3: continue

                coords = [(int(p["x"]), int(p["y"])) for p in points]
                try:
                    # 1. Create and Validate Polygon
                    bubble_polygon = Polygon(coords)
                    if not bubble_polygon.is_valid:
                        bubble_polygon = make_valid(bubble_polygon)
                        if bubble_polygon.geom_type == 'MultiPolygon':
                            bubble_polygon = max(bubble_polygon.geoms, key=lambda p: p.area, default=None)
                    if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty:
                        print(f"⚠️ Skipping bubble {i+1}: Invalid/empty polygon.")
                        continue

                    # 2. Crop *Original* Image for Translation Context
                    minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                    minx, miny = max(0, minx), max(0, miny)
                    maxx, maxy = min(w_img, maxx), min(h_img, maxy)
                    if maxx <= minx or maxy <= miny: continue

                    # Add a small buffer around the crop bounds for potentially better context
                    crop_buffer_px = 5
                    minx_c = max(0, minx - crop_buffer_px)
                    miny_c = max(0, miny - crop_buffer_px)
                    maxx_c = min(w_img, maxx + crop_buffer_px)
                    maxy_c = min(h_img, maxy + crop_buffer_px)

                    bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                    if bubble_crop.size == 0: continue

                    _, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop)
                    if crop_buffer_enc is None: continue
                    crop_bytes = crop_buffer_enc.tobytes()

                    # 3. Get Translation
                    translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, sid=sid) # Pass sid if needed by ask_luminai internally
                    if not translation:
                        print(f"⚠️ No translation for bubble {i+1}.")
                        translation = "[ترجمة فارغة]" # Placeholder for table

                    # --- Mode-Specific Actions ---
                    if mode == 'extract':
                        translations_list.append({'id': i + 1, 'translation': translation})
                        processed_count += 1

                    elif mode == 'auto':
                        if not translation or translation == "[ترجمة فارغة]":
                            print(f"⚠️ Skipping drawing for bubble {i+1} due to empty translation.")
                            continue # Don't draw if translation failed

                        # 4. Format Arabic Text (using text_formatter module)
                        arabic_text = text_formatter.format_arabic_text(translation)
                        if not arabic_text: continue

                        # 5. Shrink Polygon for Text Fitting (Initial guess)
                        poly_width = maxx - minx
                        poly_height = maxy - miny
                        initial_buffer = max(3.0, (poly_width + poly_height) / 2 * 0.08) # Adjust buffer %
                        try:
                            text_poly = bubble_polygon.buffer(-initial_buffer, join_style=2)
                            if not text_poly.is_valid or text_poly.is_empty or text_poly.geom_type != 'Polygon':
                                text_poly = bubble_polygon.buffer(-2.0, join_style=2) # Fallback shrink
                            if not isinstance(text_poly, Polygon) or not text_poly.is_valid or text_poly.is_empty:
                                print(f"⚠️ Using original bubble bounds for text fitting (bubble {i+1}).")
                                text_poly = bubble_polygon # Use original as last resort
                        except Exception:
                             print(f"⚠️ Error shrinking polygon, using original bounds (bubble {i+1}).")
                             text_poly = bubble_polygon

                        # 6. Find Optimal Text Settings
                        text_settings = find_optimal_text_settings_final(
                            temp_draw_for_settings, # Used for measurement
                            arabic_text,
                            text_poly
                        )

                        # 7. Draw Text on Layer and Composite
                        if text_settings:
                            text_layer = draw_text_on_layer(text_settings, image_size)
                            image_pil.paste(text_layer, (0, 0), text_layer)
                            processed_count += 1
                        else:
                            print(f"⚠️ Could not fit text for bubble {i+1}.")

                except Exception as bubble_err:
                    emit_error(f"Error processing bubble {i + 1}: {bubble_err}", sid)
                    # Continue with the next bubble if possible
                    import traceback
                    traceback.print_exc()

            # === Step 4: Finalize based on mode ===
            if mode == 'extract':
                emit_progress(4, f"Finished extracting text for {processed_count}/{bubble_count} bubbles.", 95, sid)
                final_image_np = inpainted_image # Final image is the cleaned one
                output_filename = f"{output_filename_base}_cleaned.jpg"
                final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                result_data = {
                    'mode': 'extract',
                    'imageUrl': f'/results/{output_filename}',
                    'translations': translations_list
                }
            elif mode == 'auto':
                emit_progress(4, f"Finished drawing text for {processed_count}/{bubble_count} bubbles.", 95, sid)
                # Convert final PIL image back to OpenCV format for saving
                final_image_rgb = image_pil.convert('RGB')
                final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR)
                output_filename = f"{output_filename_base}_translated.jpg"
                final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                result_data = {
                    'mode': 'auto',
                    'imageUrl': f'/results/{output_filename}'
                }

        # === Step 5: Save Final Image ===
        emit_progress(5, "Saving final image...", 98, sid)
        if final_image_np is None:
             raise RuntimeError("Final image data is missing.")
        if not final_output_path:
             raise RuntimeError("Final output path is not set.")

        save_success = cv2.imwrite(final_output_path, final_image_np)
        if not save_success:
            # Fallback save using PIL if OpenCV failed
            try:
                print("⚠️ OpenCV save failed, trying PIL fallback...")
                pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                pil_img_to_save.save(final_output_path)
                save_success = True
            except Exception as pil_save_err:
                raise IOError(f"Failed to save final image using OpenCV and PIL: {pil_save_err}") from pil_save_err

        # === Step 6: Signal Completion ===
        processing_time = time.time() - start_time
        emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
        socketio.emit('processing_complete', result_data, room=sid)


    except (ValueError, ConnectionError, RuntimeError, IOError) as e:
        # Catch specific, expected errors
        emit_error(f"Processing failed: {e}", sid)
        import traceback
        traceback.print_exc()
    except Exception as e:
        # Catch any unexpected errors
        emit_error(f"An unexpected error occurred: {e}", sid)
        import traceback
        traceback.print_exc()
    finally:
        # Clean up the originally uploaded file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Cleaned up: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ Error cleaning up file {image_path}: {cleanup_err}")


# --- Flask Routes ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    """Serves the processed image safely."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=False)

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # Add potential cleanup logic here if a task needs stopping for this SID

@socketio.on('start_processing')
def handle_start_processing(data):
    """Handles the image upload and starts the background task."""
    sid = request.sid
    print(f"Received 'start_processing' from {sid} with data: {list(data.keys())}") # Don't log file data

    # Validate input data
    if 'file' not in data or not data['file'].startswith('data:image'):
        emit_error('Invalid or missing file data.', sid)
        return
    if 'mode' not in data or data['mode'] not in ['extract', 'auto']:
         emit_error('Invalid or missing processing mode.', sid)
         return

    mode = data['mode']

    # Decode the base64 image data
    try:
        file_data_str = data['file']
        header, encoded = file_data_str.split(',', 1) # "data:image/png;base64,"
        file_extension = header.split('/')[1].split(';')[0]
        if file_extension not in ALLOWED_EXTENSIONS:
             emit_error(f'Invalid file type: {file_extension}', sid)
             return

        file_bytes = base64.b64decode(encoded)
        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        # Output filename base, extension added later based on mode
        output_filename_base = f"{unique_id}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        print(f"File saved to {upload_path}")

    except Exception as e:
        print(f"Error handling uploaded file: {e}")
        emit_error(f'Error processing uploaded file: {e}', sid)
        return

    # Start the background task using socketio.start_background_task
    print(f"Starting background task for SID {sid} (Mode: {mode})...")
    socketio.start_background_task(
        process_image_task,
        upload_path,          # Path to the saved uploaded file
        output_filename_base, # Base name for output file(s)
        mode,                 # Processing mode ('extract' or 'auto')
        sid                   # User's session ID
    )
    socketio.emit('processing_started', {'message': 'Processing started...'}, room=sid)

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    port = int(os.environ.get('PORT', 9000))
    print(f"Attempting to bind to host 0.0.0.0 and port {port}")
    # Use eventlet web server. Set debug=False for production.
    # Set log_output=True to see engineio/socketio logs if needed
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    # For production deployment (e.g., Heroku), use:
    # Procfile: web: gunicorn --worker-class eventlet -w 1 app:app
