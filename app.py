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

eventlet.monkey_patch() # Patch standard libraries for async operations

# --- IMPORT YOUR MODULE ---
import text_formatter

load_dotenv() # Load environment variables from .env

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
LUMINAI_API_KEY = os.getenv('LUMINAI_API_KEY') # You might not need this if LuminAI URL is public

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_here' # Change this!
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# Limit upload size, e.g., 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, async_mode='eventlet') # Use eventlet for async

# --- Ensure directories exist ---
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Font Setup (moved inside a function or done once at startup) ---
def setup_font():
    font_path_to_set = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf")
        if os.path.exists(potential_path):
            font_path_to_set = potential_path
        else:
            potential_path = os.path.join(".", "fonts", "66Hayah.otf")
            if os.path.exists(potential_path):
                font_path_to_set = potential_path

        if font_path_to_set:
            print(f"ℹ️ WebApp found Arabic font: {font_path_to_set}")
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Warning: WebApp could not find 'fonts/66Hayah.otf'. Using default PIL font.")
            text_formatter.set_arabic_font_path(None)
    except Exception as e:
        print(f"ℹ️ Error finding font path: {e}. Using default PIL font.")
        text_formatter.set_arabic_font_path(None)

setup_font() # Call setup once when the app starts

# --- Helper Functions (Adapt from your script) ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90

# --- Adapt your core functions ---
# (extract_translation, ask_luminai, get_polygon_orientation,
#  find_optimal_text_settings_final, draw_text_on_layer)
#
# IMPORTANT modifications:
# 1. Add `socketio.emit` calls within these functions to send progress updates.
# 2. Pass the `sid` (session ID) from the socket connection if needed to target specific users.
# 3. Handle errors gracefully and emit error messages.

def emit_progress(step, message, percentage, sid=None):
    """Helper to emit progress updates."""
    print(f"Progress ({step}): {message} ({percentage}%)") # Log progress server-side
    payload = {'step': step, 'message': message, 'percentage': percentage}
    if sid:
        socketio.emit('progress_update', payload, room=sid)
    else:
        # Be careful broadcasting if multiple users are processing
        socketio.emit('progress_update', payload)
    socketio.sleep(0.01) # Allow eventlet to process the emit

# --- Example Adaptation: ask_luminai ---
def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    # ... (your existing code for url, payload, headers) ...
    url = "https://luminai.my.id/" # Assuming this is the correct endpoint
    payload = {
        "content": prompt,
        "imageBuffer": list(image_bytes),
        "options": {"clean_output": True}
    }
    headers = { "Content-Type": "application/json", "Accept-Language": "ar"}

    for attempt in range(max_retries):
        try:
            # Emit progress *before* the potentially long call
            emit_progress(3, f"Contacting translation service (Attempt {attempt + 1})...", 50 + attempt*5, sid) # Example percentage
            response = requests.post(url, json=payload, headers=headers, timeout=30) # Increase timeout slightly for web

            if response.status_code == 200:
                result_text = response.json().get("result", "")
                translation = extract_translation(result_text.strip()) # Keep your extract_translation
                emit_progress(3, "Translation received.", 60, sid)
                return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                emit_progress(3, f"Translation rate limit hit. Retrying in {retry_after}s...", 50 + attempt*5, sid)
                socketio.sleep(retry_after) # Use socketio.sleep for async compatibility
            else:
                emit_progress('Error', f"LuminAI failed ({response.status_code}).", 0, sid)
                print(f"❌ LuminAI request failed: {response.status_code} - {response.text}")
                return "" # Don't retry on other errors immediately, maybe log?
        except RequestException as e:
            emit_progress('Error', f"Network error contacting LuminAI (Attempt {attempt+1}).", 0, sid)
            print(f"❌ Network error during LuminAI request: {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2 * (attempt + 1))
        except Exception as e:
            emit_progress('Error', f"Unexpected error during LuminAI request (Attempt {attempt+1}).", 0, sid)
            print(f"❌ Unexpected error during LuminAI request: {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2)
    return ""

# --- Main Processing Function (to be run in background) ---
def process_image_task(image_path, output_filename, translate, sid):
    """The core logic, adapted to emit progress."""
    try:
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        # Ensure 3 channels
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        result_image = image.copy()
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # === Step 1: Remove Text ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        _, buffer = cv2.imencode('.jpg', image)
        if buffer is None: raise ValueError("Failed to encode image for Roboflow text detection.")
        b64_image = base64.b64encode(buffer).decode('utf-8')

        try:
            response_text = requests.post(
                f'https://serverless.roboflow.com/text-detection-w0hkg/1?api_key={ROBOFLOW_API_KEY}',
                data=b64_image, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=30
            )
            response_text.raise_for_status()
            data_text = response_text.json()
            text_predictions = data_text.get("predictions", [])
            emit_progress(1, f"Found {len(text_predictions)} text areas. Masking...", 15, sid)

            polygons_drawn = 0
            for pred in text_predictions:
                points = pred.get("points", [])
                if len(points) >= 3:
                    polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                    try:
                        cv2.fillPoly(text_mask, [polygon_np], 255)
                        polygons_drawn += 1
                    except Exception as fill_err:
                        print(f"⚠️ Warning: Error drawing text polygon: {fill_err}")

            if np.any(text_mask):
                emit_progress(1, "Inpainting text areas...", 20, sid)
                result_image = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS)
                emit_progress(1, "Inpainting complete.", 25, sid)
            else:
                emit_progress(1, "No text found to remove.", 25, sid)
                result_image = image.copy()

        except requests.exceptions.RequestException as req_err:
             emit_progress('Error', f"Network error during text detection: {req_err}. Skipping removal.", 0, sid)
             print(f"❌ Network error during Roboflow text detection: {req_err}")
             result_image = image.copy() # Continue without removal
        except Exception as e:
             emit_progress('Error', f"Error during text detection/inpainting: {e}. Skipping removal.", 0, sid)
             print(f"❌ Error during Roboflow text detection or inpainting: {e}")
             result_image = image.copy() # Continue without removal


        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        bubble_predictions = []
        try:
            # Re-encode the *original* or *inpainted* image? Let's use inpainted for consistency.
            _, buffer_bubble = cv2.imencode('.jpg', result_image)
            if buffer_bubble is None: raise ValueError("Failed to encode image for bubble detection.")
            b64_image_bubble = base64.b64encode(buffer_bubble).decode('utf-8')

            response_bubbles = requests.post(
                f'https://outline.roboflow.com/yolo-0kqkh/2?api_key={ROBOFLOW_API_KEY}',
                data=b64_image_bubble, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=30
            )
            response_bubbles.raise_for_status()
            data_bubbles = response_bubbles.json()
            bubble_predictions = data_bubbles.get("predictions", [])
            emit_progress(2, f"Found {len(bubble_predictions)} speech bubbles.", 40, sid)

        except requests.exceptions.RequestException as req_err:
            emit_progress('Error', f"Network error during bubble detection: {req_err}. Cannot proceed.", 0, sid)
            raise # Re-raise to stop processing here
        except Exception as e:
            emit_progress('Error', f"Error during bubble detection: {e}. Cannot proceed.", 0, sid)
            raise # Re-raise

        if not bubble_predictions:
            emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
            # Save the cleaned (inpainted) image
            final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
            cv2.imwrite(final_output_path, result_image)
            emit_progress(5, "Processing complete.", 100, sid)
            socketio.emit('processing_complete', {'result_url': f'/results/{output_filename}'}, room=sid)
            return # Finished early

        # === Step 3: Translate and Draw (if requested) ===
        if translate:
            emit_progress(3, "Preparing for translation...", 45, sid)
            try:
                image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
            except Exception as pil_conv_err:
                emit_progress('Error', f"Failed to convert image for drawing: {pil_conv_err}", 0, sid)
                raise

            image_size = image_pil.size
            temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', image_size))
            bubble_count = len(bubble_predictions)
            processed_count = 0
            base_progress = 50 # Starting percentage for this step

            for i, pred in enumerate(bubble_predictions):
                current_bubble_progress = base_progress + int((i / bubble_count) * 40) # Scale progress within this step
                emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)

                points = pred.get("points", [])
                if len(points) < 3: continue # Skip invalid

                coords = [(int(p["x"]), int(p["y"])) for p in points]
                try:
                    original_polygon = Polygon(coords)
                    # ... (Validation logic using make_valid - same as your script) ...
                    if not original_polygon.is_valid:
                        original_polygon = make_valid(original_polygon)
                        # Handle MultiPolygon if necessary...
                    if not isinstance(original_polygon, Polygon) or original_polygon.is_empty:
                         print(f"⚠️ Skipping bubble {i+1}: Invalid/empty polygon after validation.")
                         continue

                    # --- Crop original image for context ---
                    minx_orig, miny_orig, maxx_orig, maxy_orig = map(int, original_polygon.bounds)
                    h_img, w_img = image.shape[:2] # Use original image dimensions
                    minx_orig, miny_orig = max(0, minx_orig), max(0, miny_orig)
                    maxx_orig, maxy_orig = min(w_img, maxx_orig), min(h_img, maxy_orig)

                    if maxx_orig <= minx_orig or maxy_orig <= miny_orig: continue
                    bubble_crop = image[miny_orig:maxy_orig, minx_orig:maxx_orig] # Crop from ORIGINAL
                    if bubble_crop.size == 0: continue

                    _, crop_buffer = cv2.imencode('.jpg', bubble_crop)
                    if crop_buffer is None: continue
                    crop_bytes = crop_buffer.tobytes()

                    # --- Get Translation ---
                    translation_prompt = 'ترجم نص المانجا هذا إلى اللغة العربية بحيث تكون الترجمة مفهومة وتوصل المعنى الى القارئ. أرجو إرجاع الترجمة فقط بين علامتي اقتباس مثل "النص المترجم". مع مراعاة النبرة والانفعالات الظاهرة في كل سطر (مثل: الصراخ، التردد، الهمس) وأن تُترجم بطريقة تُحافظ على الإيقاع المناسب للفقاعة.'
                    translation = ask_luminai(translation_prompt, crop_bytes, sid=sid) # Pass sid

                    if not translation:
                        print(f"⚠️ Skipping bubble {i+1}: No translation received.")
                        continue

                    # --- Shrink Polygon & Format Text (Use your existing logic) ---
                    # ... (initial shrinking logic) ...
                    initial_buffer_distance = max(3.0, (maxx_orig - minx_orig + maxy_orig - miny_orig) / 2 * 0.10) # Example
                    try:
                       initial_shrunk_polygon = original_polygon.buffer(-initial_buffer_distance, join_style=2)
                       # ... (validation and fallback for shrunk polygon) ...
                       if not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty or initial_shrunk_polygon.geom_type != 'Polygon':
                           print(f"⚠️ Using original polygon boundary for bubble {i+1} due to shrinking issues.")
                           initial_shrunk_polygon = original_polygon # Fallback
                    except Exception as buffer_err:
                        print(f"⚠️ Error shrinking polygon for bubble {i+1}: {buffer_err}. Using original.")
                        initial_shrunk_polygon = original_polygon

                    if not isinstance(initial_shrunk_polygon, Polygon) or initial_shrunk_polygon.is_empty:
                        continue

                    arabic_text = text_formatter.format_arabic_text(translation)
                    if not arabic_text: continue

                    # --- Find Optimal Settings ---
                    emit_progress(3, f"Finding layout for bubble {i + 1}...", current_bubble_progress + 5, sid)
                    text_settings = find_optimal_text_settings_final(
                        temp_draw_for_settings,
                        arabic_text,
                        initial_shrunk_polygon
                    )

                    # --- Draw and Composite ---
                    if text_settings:
                        emit_progress(3, f"Drawing text for bubble {i + 1}...", current_bubble_progress + 8, sid)
                        text_layer = draw_text_on_layer(text_settings, image_size)
                        image_pil.paste(text_layer, (0, 0), text_layer)
                        processed_count += 1
                    else:
                         print(f"⚠️ Skipping bubble {i+1}: Could not fit text.")

                except Exception as bubble_proc_err:
                    emit_progress('Error', f"Error processing bubble {i + 1}: {bubble_proc_err}", 0, sid)
                    print(f"❌ Error processing bubble {i + 1}: {bubble_proc_err}")
                    import traceback
                    traceback.print_exc()
                    # Decide whether to continue with the next bubble or stop

            emit_progress(3, f"Finished processing {processed_count}/{bubble_count} bubbles.", 90, sid)
            # Convert back to OpenCV format for saving
            final_image_rgb = image_pil.convert('RGB')
            result_image = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR)

        # === Step 4: Save Final Image ===
        emit_progress(4, "Saving final image...", 95, sid)
        final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        save_success = cv2.imwrite(final_output_path, result_image)

        if not save_success:
             # Try PIL as fallback
             try:
                 print("OpenCV save failed, trying PIL...")
                 if 'image_pil' in locals(): # If translation happened
                     final_image_rgb = image_pil.convert('RGB')
                     final_image_rgb.save(final_output_path)
                     save_success = True
                 else: # If only cleaning happened
                     pil_img_to_save = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                     pil_img_to_save.save(final_output_path)
                     save_success = True
             except Exception as pil_save_err:
                 print(f"❌ PIL save also failed: {pil_save_err}")
                 raise IOError("Failed to save the final image.") # Raise error if saving fails

        emit_progress(5, "Processing complete.", 100, sid)
        # Emit completion event with the URL to the result
        socketio.emit('processing_complete', {'result_url': f'/results/{output_filename}'}, room=sid)

    except Exception as e:
        emit_progress('Error', f"An error occurred: {e}", 0, sid)
        print(f"❌❌❌ Processing failed for sid {sid}: {e}")
        import traceback
        traceback.print_exc()
        # Emit an error event to the client
        socketio.emit('processing_error', {'error': str(e)}, room=sid)
    finally:
        # Clean up the uploaded file
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Cleaned up uploaded file: {image_path}")
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
    """Serves the processed image."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    # You could potentially send a welcome message or initial state
    # emit('status', {'message': 'Connected! Ready to process.'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # Add any cleanup logic if needed when a user disconnects mid-process

@socketio.on('start_processing')
def handle_start_processing(data):
    """Handles the image upload and starts the background task."""
    sid = request.sid # Get the unique session ID for this client
    print(f"Received start_processing event from {sid}")

    if 'file' not in data:
        emit('processing_error', {'error': 'No file part in the message'}, room=sid)
        return
    if 'translate' not in data:
         emit('processing_error', {'error': 'Missing translate option'}, room=sid)
         return

    # Decode the base64 image data
    try:
        file_data_str = data['file']
        # Split header "data:image/png;base64," from data
        header, encoded = file_data_str.split(',', 1)
        file_extension = header.split('/')[1].split(';')[0] # e.g., 'png'
        if file_extension not in ALLOWED_EXTENSIONS:
             emit('processing_error', {'error': 'Invalid file type'}, room=sid)
             return

        file_bytes = base64.b64decode(encoded)
        # Generate a unique filename
        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename = f"{unique_id}_processed.jpg" # Always save result as JPG? Or keep original ext?
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        print(f"File saved to {upload_path}")

    except Exception as e:
        print(f"Error handling uploaded file: {e}")
        emit('processing_error', {'error': f'Error processing uploaded file: {e}'}, room=sid)
        return

    translate_flag = data.get('translate', False) # Get the boolean flag

    # Start the background task using socketio.start_background_task
    # This is crucial so the socket connection doesn't block/timeout
    print(f"Starting background task for {sid}...")
    socketio.start_background_task(
        process_image_task,
        upload_path,
        output_filename,
        translate_flag,
        sid # Pass the session ID to the task
    )
    emit('processing_started', {'message': 'Processing started...'}, room=sid)


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # Use eventlet web server recommended for Flask-SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    # When deploying, use a proper WSGI server like Gunicorn with eventlet workers:
    # gunicorn --worker-class eventlet -w 1 app:app
