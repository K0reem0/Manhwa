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
    # Define dummy class *only* if the import fails.
    class DummyTextFormatter:
        """A fallback class if text_formatter module fails to import."""
        def __init__(self):
            print("⚠️ WARNING: Initializing DummyTextFormatter. Real text formatting will NOT work.")
            self._font_path = None
        def set_arabic_font_path(self, path):
            print(f"   (Dummy) Ignoring font path: {path}")
            self._font_path = path
        def get_font(self, size):
            print(f"   (Dummy) Attempting to get font size {size}.")
            try:
                from PIL import ImageFont
                try:
                    if self._font_path and os.path.exists(self._font_path): return ImageFont.truetype(self._font_path, size)
                    return ImageFont.load_default()
                except IOError: return None
            except ImportError: return None
        def format_arabic_text(self, text): return text
        def layout_balanced_text(self, draw, text, font, target_width): return text
    # Assign the dummy object if import failed
    text_formatter = DummyTextFormatter()
    print("⚠️ WARNING: Using dummy 'text_formatter' due to import error.")

load_dotenv() # Load environment variables from .env

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_default_fallback_secret_key_CHANGE_ME')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit upload size
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=False, engineio_logger=False)

# --- Ensure directories exist ---
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Uploads directory '{UPLOAD_FOLDER}' verified/created.")
    print(f"✔️ Results directory '{RESULT_FOLDER}' verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR: Could not create directories. Check permissions. Error: {e}")

# --- Font Setup ---
def setup_font():
    """Finds the font file path and sets it using the text_formatter object."""
    font_path_to_set = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(script_dir, "fonts", "66Hayah.otf") # Adjust font name if different
        if os.path.exists(potential_path):
            font_path_to_set = potential_path
        elif os.path.exists(os.path.join(".", "fonts", "66Hayah.otf")):
             font_path_to_set = os.path.join(".", "fonts", "66Hayah.otf")

        if font_path_to_set:
            print(f"ℹ️ Font found: '{font_path_to_set}'. Setting path via text_formatter.")
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found. Using default font setting via text_formatter.")
            text_formatter.set_arabic_font_path(None)
    except Exception as e:
        print(f"❌ Error during font *path finding*: {e}. Using default font setting.")
        try: text_formatter.set_arabic_font_path(None)
        except Exception as E2: print(f"❌ Error setting font path to None after another error: {E2}")

setup_font() # Call setup once at startup

# --- Constants ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا". حافظ على المعنى والنبرة الأصلية.'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)

def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=35):
    """Calls a Roboflow inference endpoint and returns predictions."""
    print(f"ℹ️ Calling Roboflow: {endpoint_url[:50]}...")
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
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        predictions = data.get("predictions", [])
        print(f"✔️ Roboflow response received. Predictions found: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err:
         print(f"❌ Roboflow Timeout Error ({endpoint_url[:50]}): {timeout_err}")
         raise ConnectionError(f"Roboflow API timed out.") from timeout_err
    except requests.exceptions.HTTPError as http_err:
         print(f"❌ Roboflow HTTP Error ({endpoint_url[:50]}): Status {http_err.response.status_code} - {http_err.response.text[:100]}")
         raise ConnectionError(f"Roboflow API request failed (Status {http_err.response.status_code}). Check API Key/Model URL.") from http_err
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Roboflow Request Error ({endpoint_url[:50]}): {req_err}")
        raise ConnectionError(f"Network error contacting Roboflow.") from req_err
    except Exception as e:
        print(f"❌ Roboflow Unexpected Error ({endpoint_url[:50]}): {e}")
        traceback.print_exc() # Log unexpected errors fully
        raise RuntimeError(f"Unexpected error during Roboflow request.") from e

def extract_translation(text):
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    if match: return match.group(1).strip()
    # Fallback: Trim quotes and whitespace if no quoted text found
    return text.strip().strip('"').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    """Sends request to LuminAI and extracts translation."""
    print("ℹ️ Calling LuminAI...")
    url = "https://luminai.my.id/" # Public endpoint
    payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}

    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30) # Increased timeout slightly
            if response.status_code == 200:
                result_text = response.json().get("result", "")
                translation = extract_translation(result_text.strip())
                print(f"✔️ LuminAI translation received: '{translation[:50]}...'")
                return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"   ⚠️ LuminAI Rate limit (429). Retrying in {retry_after}s...")
                socketio.sleep(retry_after) # Use socketio.sleep for async compatibility
            else:
                # Log non-retryable errors but return empty string
                print(f"   ❌ LuminAI request failed: Status {response.status_code} - {response.text[:100]}")
                return "" # Don't retry on other specific errors
        except requests.exceptions.Timeout:
             print(f"   ❌ LuminAI request timed out (Attempt {attempt+1}).")
             if attempt == max_retries - 1: return "" # Return empty after last retry
             socketio.sleep(2 * (attempt + 1)) # Exponential backoff might be better
        except RequestException as e:
            print(f"   ❌ Network error during LuminAI request (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1: return ""
            socketio.sleep(2 * (attempt + 1))
        except Exception as e:
            print(f"   ❌ Unexpected error during LuminAI request (Attempt {attempt+1}): {e}")
            traceback.print_exc()
            if attempt == max_retries - 1: return ""
            socketio.sleep(2)
    print("   ❌ LuminAI failed after all retries.")
    return "" # Return empty if all retries fail

# ==============================================================================
# == PLACEHOLDER FUNCTIONS - REPLACE WITH YOUR ACTUAL WORKING CODE          ====
# ==============================================================================

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """
    *** PLACEHOLDER ***
    Replace this with your actual logic to find the best font size,
    wrap the text, and calculate the drawing position (x, y) to fit
    inside the initial_shrunk_polygon. Should use text_formatter.get_font.
    Must return a dictionary like:
    {'text': wrapped_text, 'font': font_object, 'x': draw_x, 'y': draw_y, 'font_size': font_size}
    or None if no fit is found.
    """
    print("⚠️ WARNING: Using PLACEHOLDER for find_optimal_text_settings_final.")
    # --- Basic Placeholder Logic ---
    if not text or not initial_shrunk_polygon or not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty:
        return None
    try:
        font_size = 20 # Fixed size for placeholder
        font = text_formatter.get_font(font_size)
        if font is None:
            print("   ❌ Placeholder Error: Could not load font for size", font_size)
            return None # Font loading failed

        minx, miny, maxx, maxy = initial_shrunk_polygon.bounds
        target_width = maxx - minx
        if target_width <= 10: return None # Too small

        # Very basic wrapping (replace with your layout_balanced_text or similar)
        wrapped_text = text # No wrapping in placeholder
        if hasattr(text_formatter, 'layout_balanced_text'):
            wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
            if not wrapped_text: wrapped_text = text # Fallback

        # Simplified position calculation
        draw_x = int(minx) + 5
        draw_y = int(miny) + 5

        print(f"   Placeholder settings: Size={font_size}, Pos=({draw_x},{draw_y})")
        return {'text': wrapped_text, 'font': font, 'x': draw_x, 'y': draw_y, 'font_size': font_size}
    except Exception as e:
        print(f"❌ Error in (placeholder) find_optimal_text_settings_final: {e}")
        traceback.print_exc(limit=1)
        return None

def draw_text_on_layer(text_settings, image_size):
    """
    *** PLACEHOLDER ***
    Replace this with your actual PIL code to draw the formatted text
    (using text_settings['text'] and text_settings['font']) onto a new
    transparent RGBA layer of the given image_size. Include the logic
    for drawing the shadow.
    Must return the PIL Image object for the text layer.
    """
    print("⚠️ WARNING: Using PLACEHOLDER for draw_text_on_layer.")
    # --- Basic Placeholder Logic ---
    try:
        text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
        if not text_settings: return text_layer # Return empty layer if no settings

        draw_on_layer = ImageDraw.Draw(text_layer)
        font = text_settings['font']
        text_to_draw = text_settings['text']
        x, y = text_settings['x'], text_settings['y']
        font_size = text_settings['font_size']

        # Simplified drawing (replace with your shadow + text logic)
        shadow_offset = max(1, font_size // 18)
        shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)
        draw_on_layer.multiline_text((x + shadow_offset, y + shadow_offset), text_to_draw, font=font, fill=shadow_color_with_alpha, align='center', spacing=4)
        draw_on_layer.multiline_text((x, y), text_to_draw, font=font, fill=TEXT_COLOR + (255,), align='center', spacing=4)

        print(f"   Placeholder: Drew text '{text_to_draw[:20]}...' at ({x},{y})")
        return text_layer
    except Exception as e:
        print(f"❌ Error in (placeholder) draw_text_on_layer: {e}")
        traceback.print_exc(limit=1)
        # Return an empty layer on error to avoid breaking compositing
        return Image.new('RGBA', image_size, (0, 0, 0, 0))

# ==============================================================================
# == End of PLACEHOLDER Functions                                           ====
# ==============================================================================


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
        # ... (image loading/validation as before) ...
        if image is None: raise ValueError(f"Could not load image at {image_path}")
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError("Unsupported image channel format")
        h_img, w_img = image.shape[:2]
        original_image_for_cropping = image.copy()
        result_image = image.copy()


        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        # ... (encode image to b64 as before) ...
        _, buffer = cv2.imencode('.jpg', image); b64_image = base64.b64encode(buffer).decode('utf-8') if buffer is not None else None
        if not b64_image: raise ValueError("Failed to encode image for text detection.")

        text_predictions = []
        try:
            text_predictions = get_roboflow_predictions(
                'https://serverless.roboflow.com/text-detection-w0hkg/1', # Verify model ID
                ROBOFLOW_API_KEY, b64_image
            )
        except Exception as rf_err:
            print(f"⚠️ Roboflow text detection failed: {rf_err}. Proceeding without inpainting.")
            emit_progress(1, f"Text detection failed ({type(rf_err).__name__}), skipping removal.", 15, sid)

        emit_progress(1, f"Text Masking (Found {len(text_predictions)} areas)...", 15, sid)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # ... (mask drawing loop - make sure it clips coordinates) ...
        polygons_drawn = 0
        for pred in text_predictions:
             points = pred.get("points", [])
             if len(points) >= 3:
                 polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                 polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1) # Clip X
                 polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1) # Clip Y
                 try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                 except Exception as fill_err: print(f"⚠️ Warn: Error drawing text poly: {fill_err}")

        if np.any(text_mask) and polygons_drawn > 0:
            emit_progress(1, f"Inpainting {polygons_drawn} text areas...", 20, sid)
            try:
                 inpainted_image = cv2.inpaint(result_image, text_mask, 5, cv2.INPAINT_TELEA) # Or INPAINT_NS
                 if inpainted_image is None: raise RuntimeError("cv2.inpaint returned None")
                 emit_progress(1, "Inpainting complete.", 25, sid)
            except Exception as inpaint_err:
                 print(f"❌ Error during inpainting: {inpaint_err}. Using original image.")
                 emit_error(f"Inpainting failed: {inpaint_err}", sid)
                 inpainted_image = result_image.copy() # Fallback
        else:
            emit_progress(1, "No text found/masked to remove.", 25, sid)
            inpainted_image = result_image.copy()


        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        # ... (encode inpainted image to b64) ...
        _, buffer_bubble = cv2.imencode('.jpg', inpainted_image); b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8') if buffer_bubble is not None else None
        if not b64_bubble: raise ValueError("Failed to encode inpainted image for bubble detection.")

        bubble_predictions = []
        try:
             bubble_predictions = get_roboflow_predictions(
                  'https://outline.roboflow.com/yolo-0kqkh/2', # Verify model ID
                  ROBOFLOW_API_KEY, b64_bubble
             )
        except Exception as rf_err:
             print(f"❌ Bubble detection failed: {rf_err}. Aborting bubble processing.")
             # Don't raise here, just finish with the cleaned image if possible
             emit_error(f"Bubble detection failed ({type(rf_err).__name__}). Cannot translate/draw.", sid)
             bubble_predictions = [] # Ensure it's empty

        emit_progress(2, f"Bubble Detection (Found {len(bubble_predictions)} bubbles)...", 40, sid)

        # === Step 3 & 4: Process Bubbles & Finalize ===
        if not bubble_predictions:
             emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             # Always return 'cleaned' mode data if no bubbles, regardless of input mode
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        else:
            # --- Bubble processing loop ---
            image_pil = None
            image_size = (w_img, h_img) # Get size from original image dims
            temp_draw_for_settings = None
            if mode == 'auto':
                 try:
                      image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                      image_size = image_pil.size
                      # Create a temporary drawing context for measurements ONLY
                      # Avoid creating large dummy images repeatedly if possible
                      temp_img_for_measure = Image.new('RGBA', (1, 1))
                      temp_draw_for_settings = ImageDraw.Draw(temp_img_for_measure)
                 except Exception as pil_conv_err:
                      emit_error(f"Failed to prepare image for drawing: {pil_conv_err}", sid)
                      # Fallback: Continue in extract mode? Or fail? Let's try extract.
                      print(f"⚠️ Warning: PIL conversion failed, cannot perform auto-draw. Falling back.")
                      mode = 'extract' # Force extract mode if PIL fails

            bubble_count = len(bubble_predictions)
            processed_count = 0
            base_progress = 45
            max_progress_bubbles = 90

            for i, pred in enumerate(bubble_predictions):
                 current_bubble_progress = base_progress + int((i / bubble_count) * (max_progress_bubbles - base_progress))
                 emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                 try:
                    # ... (Get points, validate polygon as before) ...
                    points = pred.get("points", [])
                    if len(points) < 3: continue
                    coords = [(int(p["x"]), int(p["y"])) for p in points]
                    bubble_polygon = Polygon(coords)
                    if not bubble_polygon.is_valid: bubble_polygon = make_valid(bubble_polygon)
                    if bubble_polygon.geom_type == 'MultiPolygon': bubble_polygon = max(bubble_polygon.geoms, key=lambda p: p.area, default=None)
                    if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty: continue

                    # ... (Crop original image as before) ...
                    minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                    minx_c = max(0, minx - 5); miny_c = max(0, miny - 5)
                    maxx_c = min(w_img, maxx + 5); maxy_c = min(h_img, maxy + 5)
                    if maxx_c <= minx_c or maxy_c <= miny_c: continue
                    bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                    if bubble_crop.size == 0: continue
                    _, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop)
                    if crop_buffer_enc is None: continue
                    crop_bytes = crop_buffer_enc.tobytes()


                    # ... (Get translation as before) ...
                    translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, sid=sid)
                    if not translation: translation = "[ترجمة فارغة]"

                    # --- Mode Specific Actions ---
                    if mode == 'extract':
                         translations_list.append({'id': i + 1, 'translation': translation})
                         processed_count += 1
                    elif mode == 'auto' and image_pil: # Check image_pil exists
                         if translation == "[ترجمة فارغة]": continue

                         # --- Ensure text_formatter is usable ---
                         if isinstance(text_formatter, DummyTextFormatter):
                              print("⚠️ Skipping text drawing because DummyTextFormatter is active.")
                              continue # Cannot format/draw without real formatter

                         # --- Format, Layout, Draw ---
                         arabic_text = text_formatter.format_arabic_text(translation)
                         if not arabic_text: continue

                         # ... (Shrink polygon as before) ...
                         initial_buffer = max(3.0, (maxx - minx + maxy - miny) / 2 * 0.08)
                         text_poly = bubble_polygon # Default
                         try:
                              shrunk = bubble_polygon.buffer(-initial_buffer, join_style=2)
                              if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                              else: # Try smaller shrink on failure
                                   shrunk = bubble_polygon.buffer(-2.0, join_style=2)
                                   if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type == 'Polygon': text_poly = shrunk
                         except Exception: pass # Ignore buffer errors

                         # --- Call the (potentially placeholder) layout function ---
                         text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                         if text_settings:
                              # --- Call the (potentially placeholder) drawing function ---
                              text_layer = draw_text_on_layer(text_settings, image_size)
                              if text_layer:
                                   try:
                                       image_pil.paste(text_layer, (0, 0), text_layer)
                                       processed_count += 1
                                   except Exception as paste_err:
                                       print(f"❌ Error pasting text layer for bubble {i+1}: {paste_err}")
                              else: print(f"⚠️ Text layer drawing failed for bubble {i+1}")
                         else: print(f"⚠️ Could not fit text for bubble {i+1}.")

                 except Exception as bubble_err:
                      print(f"❌ Error processing bubble {i + 1}: {bubble_err}\n---")
                      traceback.print_exc(limit=1)
                      emit_progress(3, f"Skipping bubble {i+1} due to error.", current_bubble_progress + 1, sid)

            # --- Finalize based on mode after loop ---
            if mode == 'extract':
                 emit_progress(4, f"Finished extracting text ({processed_count}/{bubble_count}).", 95, sid)
                 final_image_np = inpainted_image
                 output_filename = f"{output_filename_base}_cleaned.jpg"
                 final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                 result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
            elif mode == 'auto': # Should only be reached if image_pil exists
                 emit_progress(4, f"Finished drawing text ({processed_count}/{bubble_count}).", 95, sid)
                 if image_pil:
                      final_image_rgb = image_pil.convert('RGB')
                      final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR)
                 else: # Should not happen if logic above is correct, but fallback
                      final_image_np = inpainted_image
                 output_filename = f"{output_filename_base}_translated.jpg"
                 final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
                 result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}


        # === Step 5: Save Final Image ===
        emit_progress(5, "Saving final image...", 98, sid)
        # ... (save logic with PIL fallback as before) ...
        if final_image_np is None: raise RuntimeError("Final image data is missing before save.")
        if not final_output_path: raise RuntimeError("Final output path is not set before save.")
        save_success = cv2.imwrite(final_output_path, final_image_np)
        if not save_success:
             try:
                  print("⚠️ OpenCV save failed, trying PIL fallback...")
                  pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                  pil_img_to_save.save(final_output_path)
             except Exception as pil_save_err:
                  raise IOError(f"Failed to save final image using OpenCV and PIL: {pil_save_err}") from pil_save_err

        # === Step 6: Signal Completion ===
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Output: {final_output_path}")
        emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
        socketio.emit('processing_complete', result_data, room=sid)

    # --- Main Task Error Handler ---
    except Exception as e:
        print(f"❌❌❌ Unhandled error in process_image_task for SID {sid}: {e}")
        traceback.print_exc()
        emit_error(f"An unexpected server error occurred: {type(e).__name__}. Check server logs.", sid)
    finally:
        # --- Cleanup Uploaded File ---
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Cleaned up input file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ Error cleaning up file {image_path}: {cleanup_err}")


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results/<filename>')
def get_result_image(filename):
    if '..' in filename or filename.startswith('/'): return "Invalid filename", 400
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
    # --- This handler remains the same robust version from the previous step ---
    # --- It validates input, directories, saves the file, and starts the task ---
    sid = request.sid
    print(f"\n--- Received 'start_processing' event from SID: {sid} ---")

    if not isinstance(data, dict): # Basic Type Check
        print(f"   ❗ ERROR: Invalid data type received from {sid}. Expected dict.")
        emit_error("Invalid request data format.", sid); return
    print(f"   Data keys received: {list(data.keys())}")
    if 'file' not in data or not isinstance(data['file'], str) or not data['file'].startswith('data:image'):
        print(f"   ❗ ERROR: Missing or invalid 'file' data from {sid}.")
        emit_error('Invalid or missing file data.', sid); return
    if 'mode' not in data or data['mode'] not in ['extract', 'auto']:
        print(f"   ❗ ERROR: Missing or invalid 'mode' data from {sid}.")
        emit_error('Invalid or missing processing mode.', sid); return
    mode = data['mode']
    print(f"   Mode validated: '{mode}'")

    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir) or not os.access(upload_dir, os.W_OK):
        print(f"   ❗ SERVER ERROR: Upload directory '{upload_dir}' issue (Exists: {os.path.exists(upload_dir)}, Writable: {os.access(upload_dir, os.W_OK)})")
        emit_error("Server configuration error (upload dir).", sid); return
    print(f"   Upload directory '{upload_dir}' checked.")

    upload_path = None
    try:
        print(f"   Decoding Base64 data...")
        file_data_str = data['file']
        header, encoded = file_data_str.split(',', 1)
        file_extension = header.split('/')[1].split(';')[0].split('+')[0]
        if file_extension not in ALLOWED_EXTENSIONS:
             print(f"   ❗ ERROR: Invalid file extension '{file_extension}' from {sid}.")
             emit_error(f'Invalid file type: {file_extension}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', sid); return

        file_bytes = base64.b64decode(encoded)
        print(f"   Base64 decoded successfully. Size: {len(file_bytes) / 1024:.1f} KB")

        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename_base = f"{unique_id}"
        upload_path = os.path.join(upload_dir, input_filename)

        print(f"   Attempting to write file to: {upload_path}")
        with open(upload_path, 'wb') as f: f.write(file_bytes)
        print(f"   ✔️ File successfully saved.")

    except (base64.binascii.Error, ValueError) as decode_err:
        print(f"   ❗ ERROR: Base64 decoding failed for {sid}: {decode_err}"); emit_error(f"Failed to decode image data: {decode_err}", sid); return
    except OSError as write_err:
         print(f"   ❗ ERROR: Failed to write file '{upload_path}' for {sid}: {write_err}"); traceback.print_exc(); emit_error(f"Server error saving file: {write_err}", sid); return
    except Exception as e:
        print(f"   ❗ UNEXPECTED ERROR during file handling for {sid}: {e}"); traceback.print_exc(); emit_error(f'Unexpected server error during file upload: {type(e).__name__}', sid); return

    if upload_path:
        print(f"   Attempting to start background task (Mode: '{mode}') for: {upload_path}")
        try:
            socketio.start_background_task(process_image_task, upload_path, output_filename_base, mode, sid)
            print(f"   ✔️ Background task initiated for SID: {sid}")
            socketio.emit('processing_started', {'message': 'Upload successful! Processing started...'}, room=sid)
        except Exception as task_err:
            print(f"   ❗ CRITICAL ERROR: Failed to start background task for {sid}: {task_err}"); traceback.print_exc(); emit_error(f"Server error initiating processing task: {task_err}", sid)
            if os.path.exists(upload_path):
                try: os.remove(upload_path); print(f"   🧹 Cleaned up file due to task start failure: {upload_path}")
                except Exception: print(f"   ⚠️ Could not clean up file after task start failure: {upload_path}")
    else:
         print(f"   ❗ LOGIC ERROR: upload_path not set, cannot start background task for {sid}"); emit_error("Internal server error (upload path missing).", sid)

    print(f"--- Finished handling 'start_processing' event for SID: {sid} ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Flask-SocketIO Application ---")
    if not ROBOFLOW_API_KEY: print("⚠️ WARNING: ROBOFLOW_API_KEY environment variable is not set!")
    if app.config['SECRET_KEY'] == 'a_default_fallback_secret_key_CHANGE_ME': print("⚠️ WARNING: Using default FLASK_SECRET_KEY!")
    port = int(os.environ.get('PORT', 9000))
    print(f"   * Environment: {os.environ.get('FLASK_ENV', 'production')}")
    print(f"   * Binding to: host 0.0.0.0, port {port}")
    print(f"   * Async mode: eventlet")
    print(f"   * CORS Allowed Origins: *")
    print("--- Ready for connections ---")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
