:# -*- coding: utf-8 -*-
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
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback

eventlet.monkey_patch()

# --- IMPORT YOUR MODULE ---
try:
    import text_formatter # Assuming text_formatter.py is in the same directory
    print("✔️ Successfully imported 'text_formatter.py'.")
    # Check if required functions exist
    if not all(hasattr(text_formatter, func) for func in ['set_arabic_font_path', 'get_font', 'format_arabic_text', 'layout_balanced_text']):
         print("⚠️ WARNING: 'text_formatter.py' seems to be missing required functions!")
         raise ImportError("Missing functions in text_formatter")
except ImportError as import_err:
    print(f"❌ ERROR: Cannot import 'text_formatter.py' or it's incomplete: {import_err}")
    class DummyTextFormatter:
        def __init__(self): print("⚠️ WARNING: Initializing DummyTextFormatter.")
        def set_arabic_font_path(self, path): pass
        def get_font(self, size): return None
        def format_arabic_text(self, text): return text
        def layout_balanced_text(self, draw, text, font, target_width): return text
    text_formatter = DummyTextFormatter()
    print("⚠️ WARNING: Using dummy 'text_formatter'. Text formatting/layout will be basic or fail.")

load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY') # Use environment variable

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=False, engineio_logger=False)

# --- Ensure directories exist ---
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Directories '{UPLOAD_FOLDER}' and '{RESULT_FOLDER}' verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR creating directories: {e}")

# --- Font Setup ---
def setup_font():
    font_path_to_set = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf") # Your font file
        if os.path.exists(potential_path):
            font_path_to_set = potential_path
        elif os.path.exists(os.path.join(".", "fonts", "66Hayah.otf")):
             font_path_to_set = os.path.join(".", "fonts", "66Hayah.otf")

        if font_path_to_set:
            print(f"ℹ️ Font found: '{font_path_to_set}'. Setting path.")
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found. Using default.")
            text_formatter.set_arabic_font_path(None)
    except Exception as e:
        print(f"❌ Error during font path finding: {e}. Using default.")
        try: text_formatter.set_arabic_font_path(None)
        except Exception as E2: print(f"❌ Error setting font path to None: {E2}")
setup_font()

# --- Constants (from your script) ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
# Use a clear, concise prompt for the web app context
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)

def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

# --- Integrated Core Logic Functions ---

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=30):
    """Calls a Roboflow inference endpoint and returns predictions."""
    # This combines the logic from Step 1 and Step 2 of your original script
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown Model"
    print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key:
         print("❌ ERROR: Roboflow API Key is missing!")
         raise ValueError("Missing Roboflow API Key.")

    try:
        response = requests.post(
            f"{endpoint_url}?api_key={api_key}",
            data=image_b64,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        predictions = data.get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err:
         print(f"❌ Roboflow ({model_name}) Timeout Error: {timeout_err}")
         raise ConnectionError(f"Roboflow API ({model_name}) timed out.") from timeout_err
    except requests.exceptions.HTTPError as http_err:
         print(f"❌ Roboflow ({model_name}) HTTP Error: Status {http_err.response.status_code}")
         try: print(f"   Response text: {http_err.response.text[:200]}") # Log part of the error response
         except Exception: pass
         raise ConnectionError(f"Roboflow API ({model_name}) request failed (Status {http_err.response.status_code}). Check Key/URL.") from http_err
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Roboflow ({model_name}) Request Error: {req_err}")
        raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e:
        print(f"❌ Roboflow ({model_name}) Unexpected Error: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Unexpected error during Roboflow ({model_name}) request.") from e

def extract_translation(text):
    """Extracts text within the first pair of double quotes."""
    # Copied directly from your script
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Fallback: strip outer quotes and whitespace if no quoted text found
        return text.strip().strip('"').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    """Sends request to LuminAI and extracts translation."""
    # Copied directly from your script, using the prompt constant
    print("ℹ️ Calling LuminAI...")
    url = "https://luminai.my.id/"
    payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}

    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30) # Slightly longer timeout
            if response.status_code == 200:
                result_text = response.json().get("result", "")
                translation = extract_translation(result_text) # Use helper here
                print(f"✔️ LuminAI translation received: '{translation[:50]}...'")
                return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"   ⚠️ LuminAI Rate limit (429). Retrying in {retry_after}s...")
                socketio.sleep(retry_after) # Use socketio.sleep for async
            else:
                print(f"   ❌ LuminAI request failed: Status {response.status_code} - {response.text[:100]}")
                return "" # Don't retry other errors
        except RequestException as e:
            print(f"   ❌ Network/Timeout error during LuminAI (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"   ❌ Unexpected error during LuminAI (Attempt {attempt+1}): {e}")
            traceback.print_exc(limit=1)
            if attempt == max_retries - 1: return ""
            socketio.sleep(2)
    print("   ❌ LuminAI failed after all retries.")
    return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """
    Searches for the best font size and text layout to fit within the polygon.
    Uses text_formatter.layout_balanced_text.
    Copied from your script - relies on text_formatter.
    """
    print("ℹ️ Finding optimal text settings...")
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid:
        print("   ⚠️ Invalid/empty polygon passed.")
        return None
    if not text:
        print("   ⚠️ Empty text passed.")
        return None

    best_fit = None
    for font_size in range(65, 4, -1): # Iterate font sizes
        font = text_formatter.get_font(font_size) # Use formatter
        if font is None:
             # Log this potentially important failure
             # print(f"   Debug: Failed to get font for size {font_size}")
             continue # Skip size if font loading fails

        # Calculate padding based on font size
        padding_distance = max(1.5, font_size * 0.12)
        try:
            # Shrink further for text fitting, with fallback
            text_fitting_polygon = initial_shrunk_polygon.buffer(-padding_distance, join_style=2)
            if not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty:
                text_fitting_polygon = initial_shrunk_polygon.buffer(-2.0, join_style=2) # Smaller fallback buffer
            # Ensure it's still a valid polygon
            if not isinstance(text_fitting_polygon, Polygon) or not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty:
                 continue
        except Exception as buffer_err:
            print(f"   ⚠️ Error buffering polygon for font size {font_size}: {buffer_err}. Skipping size.")
            continue

        minx, miny, maxx, maxy = text_fitting_polygon.bounds
        target_width = maxx - minx
        target_height = maxy - miny

        # Check if target area is reasonably sized
        if target_width <= 5 or target_height <= 10: continue

        # --- Use text_formatter's layout function ---
        try:
             wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        except Exception as layout_err:
             print(f"   ⚠️ Error in layout_balanced_text for font size {font_size}: {layout_err}. Skipping size.")
             continue # Skip if layout function fails
        if not wrapped_text: continue # Skip if layout returns empty

        try:
            # Measure the actual dimensions using Pillow's multiline_textbbox
            # Use the temporary 'draw' context passed for measurement
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_actual_width = m_bbox[2] - m_bbox[0]
            text_actual_height = m_bbox[3] - m_bbox[1]
            shadow_offset = max(1, font_size // 18) # Estimate shadow size

            # Check if text (plus shadow estimate) fits within the target dimensions
            if (text_actual_height + shadow_offset) <= target_height and \
               (text_actual_width + shadow_offset) <= target_width:
                # --- Fit found ---
                # Calculate position to center the text block
                x_offset = (target_width - text_actual_width) / 2
                y_offset = (target_height - text_actual_height) / 2
                # Adjust draw position based on the measured bbox's top-left (m_bbox[0], m_bbox[1])
                draw_x = minx + x_offset - m_bbox[0]
                draw_y = miny + y_offset - m_bbox[1]

                best_fit = {
                    'text': wrapped_text,
                    'font': font,
                    'x': int(draw_x),
                    'y': int(draw_y),
                    'font_size': font_size
                }
                print(f"   ✔️ Optimal fit found: Size={font_size}, Pos=({int(draw_x)},{int(draw_y)})")
                break # Exit loop once best fit is found
        except Exception as measure_err:
            print(f"   ⚠️ Error measuring text dimensions for size {font_size}: {measure_err}. Skipping size.")
            # Optionally add traceback.print_exc(limit=1) here for detailed debug
            continue

    if best_fit is None:
        print(f"   ⚠️ Warning: Could not find suitable font size for text: '{text[:30]}...'")

    return best_fit

def draw_text_on_layer(text_settings, image_size):
    """Draws the text with shadow onto a new transparent layer."""
    # Copied directly from your script
    print("ℹ️ Drawing text layer...")
    try:
        text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
        if not text_settings or not isinstance(text_settings, dict):
             print("   ⚠️ Invalid text_settings passed to draw_text_on_layer.")
             return text_layer # Return empty layer

        draw_on_layer = ImageDraw.Draw(text_layer)

        # Safely get values from dict
        font = text_settings.get('font')
        text_to_draw = text_settings.get('text', '')
        x = text_settings.get('x', 0)
        y = text_settings.get('y', 0)
        font_size = text_settings.get('font_size', 10) # Default size if missing

        if not font or not text_to_draw:
             print("   ⚠️ Missing font or text in text_settings for drawing.")
             return text_layer # Return empty

        shadow_offset = max(1, font_size // 18)
        shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)

        # Draw shadow first
        draw_on_layer.multiline_text(
            (x + shadow_offset, y + shadow_offset),
            text_to_draw, font=font, fill=shadow_color_with_alpha,
            align='center', spacing=4
        )
        # Draw text on top
        draw_on_layer.multiline_text(
            (x, y),
            text_to_draw, font=font, fill=TEXT_COLOR + (255,), # Use TEXT_COLOR constant
            align='center', spacing=4
        )
        print(f"   ✔️ Drew text '{text_to_draw[:20]}...' at ({x},{y})")
        return text_layer
    except Exception as e:
        print(f"❌ Error in draw_text_on_layer: {e}")
        traceback.print_exc(limit=1)
        # Return an empty layer on error
        return Image.new('RGBA', image_size, (0, 0, 0, 0))

# --- Main Processing Task ---
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
        elif image.shape[2] != 3: raise ValueError("Unsupported image format")
        h_img, w_img = image.shape[:2]
        original_image_for_cropping = image.copy()
        result_image = image.copy() # Image to potentially inpaint


        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        _, buffer = cv2.imencode('.jpg', image); b64_image = base64.b64encode(buffer).decode('utf-8') if buffer is not None else None
        if not b64_image: raise ValueError("Failed to encode image for text detection.")

        text_predictions = []
        try:
            # Use the integrated function with actual API call
            text_predictions = get_roboflow_predictions(
                'https://serverless.roboflow.com/text-detection-w0hkg/1', # Verify Model ID
                ROBOFLOW_API_KEY, b64_image
            )
        except Exception as rf_err:
            # Log error but try to continue without inpainting
            print(f"⚠️ Roboflow text detection failed: {rf_err}. Proceeding without inpainting.")
            emit_progress(1, f"Text detection failed ({type(rf_err).__name__}), skipping removal.", 15, sid)

        emit_progress(1, f"Text Masking (Found {len(text_predictions)} areas)...", 15, sid)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons_drawn = 0
        for pred in text_predictions:
             points = pred.get("points", [])
             if len(points) >= 3:
                 polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                 polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                 try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                 except Exception as fill_err: print(f"⚠️ Warn: Error drawing text poly: {fill_err}")

        if np.any(text_mask) and polygons_drawn > 0:
            emit_progress(1, f"Inpainting {polygons_drawn} text areas...", 20, sid)
            try:
                 # Use INPAINT_NS as in original script, could also try INPAINT_TELEA
                 inpainted_image = cv2.inpaint(result_image, text_mask, 10, cv2.INPAINT_NS)
                 if inpainted_image is None: raise RuntimeError("cv2.inpaint returned None")
                 emit_progress(1, "Inpainting complete.", 25, sid)
            except Exception as inpaint_err:
                 print(f"❌ Error during inpainting: {inpaint_err}. Using original image.")
                 emit_error(f"Inpainting failed: {inpaint_err}", sid); inpainted_image = result_image.copy()
        else:
            emit_progress(1, "No text found/masked to remove.", 25, sid); inpainted_image = result_image.copy()


        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        _, buffer_bubble = cv2.imencode('.jpg', inpainted_image); b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8') if buffer_bubble is not None else None
        if not b64_bubble: raise ValueError("Failed to encode inpainted image for bubble detection.")

        bubble_predictions = []
        try:
             # Use integrated function with actual API call
             bubble_predictions = get_roboflow_predictions(
                  'https://outline.roboflow.com/yolo-0kqkh/2', # Verify Model ID
                  ROBOFLOW_API_KEY, b64_bubble
             )
        except Exception as rf_err:
             print(f"❌ Bubble detection failed: {rf_err}. Cannot proceed with translation/drawing.")
             emit_error(f"Bubble detection failed ({type(rf_err).__name__}).", sid)
             # Finish with cleaned image if possible
             final_image_np = inpainted_image # Use the result of inpainting step
             output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
             # Jump directly to saving step (outside the main try...except of the task)
             # Need to signal completion differently or handle in finally block
             # For now, let the outer try-except catch this state if needed,
             # but ideally we save and emit completion here. Let's try saving:
             emit_progress(5, "Saving cleaned image (no bubbles found/detected)...", 98, sid)
             save_success = cv2.imwrite(final_output_path, final_image_np)
             # Add PIL fallback if needed
             if save_success:
                 processing_time = time.time() - start_time
                 print(f"✔️ SID {sid} Processing complete (no bubbles) in {processing_time:.2f}s.")
                 emit_progress(6, f"Processing complete (no bubbles).", 100, sid)
                 socketio.emit('processing_complete', result_data, room=sid)
             else:
                 emit_error("Failed to save intermediate cleaned image.", sid)
             return # Exit the task early


        emit_progress(2, f"Bubble Detection (Found {len(bubble_predictions)} bubbles)...", 40, sid)

        # === Step 3 & 4: Process Bubbles & Finalize ===
        if not bubble_predictions:
             # This case is now handled above if API fails, but keep for 0 detections case
             emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned.jpg"
             # ... (save and emit logic as above) ...
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        else:
             # --- Bubble processing loop ---
             image_pil = None; image_size = (w_img, h_img); temp_draw_for_settings = None
             if mode == 'auto':
                 try:
                      image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                      image_size = image_pil.size
                      temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1))) # Minimal temp obj
                 except Exception as pil_conv_err:
                      emit_error(f"Cannot draw text (Image conversion failed: {pil_conv_err})", sid)
                      print(f"⚠️ Warning: PIL conversion failed, falling back to extract mode."); mode = 'extract'

             bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90

             for i, pred in enumerate(bubble_predictions):
                 current_bubble_progress = base_progress + int((i / bubble_count) * (max_progress_bubbles - base_progress))
                 emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                 try:
                      # ... (Get points, validate polygon as before) ...
                      points = pred.get("points", [])
                      if len(points) < 3: continue
                      coords=[(int(p["x"]),int(p["y"])) for p in points]; bubble_polygon = Polygon(coords)
                      if not bubble_polygon.is_valid: bubble_polygon = make_valid(bubble_polygon)
                      if bubble_polygon.geom_type=='MultiPolygon': bubble_polygon = max(bubble_polygon.geoms, key=lambda p:p.area,default=None)
                      if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty: continue

                      # ... (Crop original image as before) ...
                      minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                      minx_c=max(0,minx-5); miny_c=max(0,miny-5); maxx_c=min(w_img,maxx+5); maxy_c=min(h_img,maxy+5)
                      if maxx_c<=minx_c or maxy_c<=miny_c: continue
                      bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                    if bubble_crop.size == 0:
                        print(f"   ⚠️ Bubble crop {i+1} is empty. Skipping.")
                        continue # Skips to next bubble

                    # --- CORRECTED Encoding Check ---
                    print(f"   Encoding bubble crop {i+1}...")
                    retval, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop) # Get return value
                    if not retval: # Check the boolean success flag
                        print(f"   ⚠️ Failed to encode bubble crop {i+1} to JPG (retval=False). Skipping.")
                        continue # Skips to the next bubble if encoding failed
                    # --- Encoding succeeded, now assign bytes ---
                    crop_bytes = crop_buffer_enc.tobytes()

                    # --- Call translation function (Now crop_bytes is guaranteed to be assigned) ---
                    translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, sid=sid)
                      if not translation: translation = "[ترجمة فارغة]"

                      # --- Mode Specific Actions ---
                      if mode == 'extract':
                           translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                      elif mode == 'auto' and image_pil:
                           if translation == "[ترجمة فارغة]": continue
                           if isinstance(text_formatter, DummyTextFormatter): print("⚠️ Skipping drawing (DummyFormatter)"); continue

                           arabic_text = text_formatter.format_arabic_text(translation)
                           if not arabic_text: continue

                           # ... (Shrink polygon as in original script) ...
                           poly_width = maxx - minx; poly_height = maxy - miny
                           initial_buffer_distance = max(3.0, (poly_width + poly_height) / 2 * 0.10) # From original
                           text_poly = bubble_polygon # Default
                           try:
                                shrunk = bubble_polygon.buffer(-initial_buffer_distance, join_style=2)
                                if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                                else:
                                     shrunk = bubble_polygon.buffer(-3.0, join_style=2) # Fallback from original
                                     if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                           except Exception as buffer_err: print(f"⚠️ Warn: buffer error {buffer_err}")

                           # --- Call ACTUAL layout function ---
                           text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                           if text_settings:
                                # --- Call ACTUAL drawing function ---
                                text_layer = draw_text_on_layer(text_settings, image_size)
                                if text_layer:
                                     try: image_pil.paste(text_layer, (0, 0), text_layer); processed_count += 1
                                     except Exception as paste_err: print(f"❌ Paste Error bbl {i+1}: {paste_err}")
                                else: print(f"⚠️ Draw func failed bbl {i+1}")
                           else: print(f"⚠️ Text fit failed bbl {i+1}")

                 except Exception as bubble_err:
                      print(f"❌ Error processing bubble {i + 1}: {bubble_err}"); traceback.print_exc(limit=1)
                      emit_progress(3, f"Skipping bubble {i+1} (error).", current_bubble_progress + 1, sid)

             # --- Finalize based on mode after loop ---
             if mode == 'extract':
                 emit_progress(4, f"Finished extracting text ({processed_count}/{bubble_count}).", 95, sid)
                 final_image_np = inpainted_image; output_filename = f"{output_filename_base}_cleaned.jpg"
                 result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
             elif mode == 'auto':
                 emit_progress(4, f"Finished drawing text ({processed_count}/{bubble_count}).", 95, sid)
                 final_image_np = inpainted_image # Default to cleaned if PIL failed
                 if image_pil: # Convert PIL back if it exists
                      try: final_image_np = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                      except Exception as convert_err: print(f"❌ Error converting PIL->CV2: {convert_err}")
                 output_filename = f"{output_filename_base}_translated.jpg"
                 result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)


        # === Step 5: Save Final Image ===
        emit_progress(5, "Saving final image...", 98, sid)
        if final_image_np is None: raise RuntimeError("Final image is None before save.")
        if not final_output_path: raise RuntimeError("Final output path not set before save.")
        save_success = False
        try:
             save_success = cv2.imwrite(final_output_path, final_image_np)
             if not save_success: raise IOError("cv2.imwrite returned false") # More explicit error
        except Exception as cv_save_err:
             print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL fallback...")
             try:
                 pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                 pil_img_to_save.save(final_output_path)
                 save_success = True
                 print("✔️ Saved successfully using PIL fallback.")
             except Exception as pil_save_err:
                  print(f"❌ PIL save also failed: {pil_save_err}")
                  raise IOError(f"Failed to save final image using OpenCV and PIL") from pil_save_err # Raise error if both fail

        # === Step 6: Signal Completion ===
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Output: {final_output_path}")
        emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
        socketio.emit('processing_complete', result_data, room=sid) # Send results

    except Exception as e:
        print(f"❌❌❌ Unhandled error in process_image_task for SID {sid}: {e}")
        traceback.print_exc()
        emit_error(f"An unexpected server error occurred ({type(e).__name__}). Please check server logs.", sid)
    finally:
        # Cleanup Uploaded File
        try:
            if image_path and os.path.exists(image_path): os.remove(image_path); print(f"🧹 Cleaned up: {image_path}")
        except Exception as cleanup_err: print(f"⚠️ Error cleaning up file {image_path}: {cleanup_err}")

# --- Flask Routes & SocketIO Handlers ---
# (These remain the same as the previous version)
@app.route('/')
def index(): return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    if '..' in filename or filename.startswith('/'): return "Invalid filename", 400
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=False)

@socketio.on('connect')
def handle_connect(): print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect(): print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    sid = request.sid
    print(f"\n--- Received 'start_processing' event from SID: {sid} ---")
    # --- Robust input handling and file saving ---
    if not isinstance(data, dict): emit_error("Invalid request format.", sid); return
    print(f"   Data keys: {list(data.keys())}")
    if 'file' not in data or not isinstance(data['file'], str) or not data['file'].startswith('data:image'): emit_error('Invalid/missing file data.', sid); return
    if 'mode' not in data or data['mode'] not in ['extract', 'auto']: emit_error('Invalid/missing mode.', sid); return
    mode = data['mode']; print(f"   Mode: '{mode}'")
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir) or not os.access(upload_dir, os.W_OK): emit_error("Server upload dir error.", sid); return
    upload_path = None # Initialize path variable
    try:
        print(f"   Decoding Base64 data...") # Added log
        file_data_str = data['file']
        # Split header and decode
        try:
            header, encoded = file_data_str.split(',', 1)
            file_extension = header.split('/')[1].split(';')[0].split('+')[0] # Handle image/svg+xml etc.
        except ValueError:
            # Handle cases where the base64 string format might be unexpected
            print(f"   ❗ ERROR: Invalid base64 header format from {sid}.")
            emit_error('Invalid image data header.', sid)
            return # Stop processing

        # Validate extension
        if file_extension not in ALLOWED_EXTENSIONS:
             print(f"   ❗ ERROR: Invalid file extension '{file_extension}' from {sid}.")
             emit_error(f'Invalid file type: {file_extension}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', sid)
             return # Stop processing

        print(f"   File extension: '{file_extension}'")
        file_bytes = base64.b64decode(encoded) # Can raise binascii.Error
        print(f"   Base64 decoded successfully. Size: {len(file_bytes) / 1024:.1f} KB")

        # Generate paths
        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename_base = f"{unique_id}" # Base name for results
        upload_path = os.path.join(upload_dir, input_filename) # Use checked upload_dir

        # Save the file
        print(f"   Attempting to write file to: {upload_path}")
        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        print(f"   ✔️ File successfully saved.")

    except (base64.binascii.Error, ValueError) as decode_err:
        # Handle errors during decoding or header splitting
        print(f"   ❗ ERROR: Base64 decoding/format failed for {sid}: {decode_err}")
        emit_error(f"Failed to decode image data: {decode_err}", sid)
        return # Stop processing
    except OSError as write_err:
         # Handle errors during file writing
         print(f"   ❗ ERROR: Failed to write file '{upload_path}' for {sid}: {write_err}")
         traceback.print_exc() # Log full traceback for server admin
         emit_error(f"Server error saving file: {write_err}", sid)
         # Ensure upload_path is None if write failed, preventing task start attempt
         upload_path = None # Explicitly set to None on write error
         return # Stop processing
    except Exception as e:
        # Catch any other unexpected errors during file handling
        print(f"   ❗ UNEXPECTED ERROR during file handling for {sid}: {e}")
        traceback.print_exc()
        emit_error(f'Unexpected server error during file upload: {type(e).__name__}', sid)
        upload_path = None # Explicitly set to None on error
        return # Stop processing

    # --- Start Task ---
    # Only attempt to start if upload_path was successfully set (file saved)
    if upload_path:
        print(f"   Attempting to start background task (Mode: '{mode}') for: {upload_path}")
        try:
            # Initiate the background processing task
            socketio.start_background_task(
                process_image_task,
                upload_path,
                output_filename_base,
                mode,
                sid
            )
            # Confirm task initiation to logs and client
            print(f"   ✔️ Background task initiated for SID: {sid}")
            socketio.emit('processing_started', {'message': 'Upload successful! Processing started...'}, room=sid)

        except Exception as task_err:
            # Handle errors specifically related to *starting* the task
            print(f"   ❗ CRITICAL ERROR: Failed to start background task for {sid}: {task_err}")
            traceback.print_exc()
            emit_error(f"Server error starting task: {task_err}", sid)

            # --- FIXED FILE CLEANUP on task start failure ---
            # This block replaces the problematic one-liner
            # Ensure it has the same indentation as the print/traceback/emit lines above
            print(f"   Attempting cleanup for failed task start: {upload_path}")
            # Check path exists before trying to remove
            if os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                    print(f"   🧹 Cleaned up file due to task start failure: {upload_path}")
                except Exception as cleanup_err:
                    # Log cleanup errors but don't crash the handler
                    print(f"   ⚠️ Could not clean up file after task start failure: {upload_path} - {cleanup_err}")
            # --- END OF FIXED FILE CLEANUP ---

    else:
        # This handles the case where upload_path remained None (due to an earlier error)
        print(f"   ❗ ERROR: Cannot start task because upload_path is not set (file save likely failed). SID: {sid}")
        # Emit error only if no previous error was emitted for the file handling part
        # (Avoids duplicate error messages to the client)
        # emit_error("Internal server error (cannot start task).", sid) # Maybe too generic

    # This print marks the end of the 'handle_start_processing' function execution path
    print(f"--- Finished handling 'start_processing' event for SID: {sid} ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App ---")
    if not ROBOFLOW_API_KEY: print("⚠️ WARNING: ROBOFLOW_API_KEY not set!")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default FLASK_SECRET_KEY!")
    port = int(os.environ.get('PORT', 9000))
    print(f"   * Ready on http://0.0.0.0:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False) # Use debug=True for more Flask/Werkzeug logs locally if needed
