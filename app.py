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
from shapely.geometry import Polygon, MultiPolygon # Keep both imports
from shapely.validation import make_valid
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename # <-- Import secure_filename

eventlet.monkey_patch()

# --- Define DummyTextFormatter Globally First ---
# (Keep DummyTextFormatter and text_formatter import logic as before)
class DummyTextFormatter:
    def __init__(self): print("⚠️ WARNING: Initializing DummyTextFormatter.")
    def set_arabic_font_path(self, path): pass
    def get_font(self, size): return None
    def format_arabic_text(self, text): return text
    def layout_balanced_text(self, draw, text, font, target_width): return text

try:
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
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# MAX_CONTENT_LENGTH still applies to the HTTP POST upload size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# --- SocketIO Setup ---
# Increased ping timeout for potentially slower connections, less critical now for upload itself
socketio = SocketIO(
    app,
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False, # Set to True for more verbose debugging if needed
    engineio_logger=False, # Set to True for more verbose debugging if needed
    ping_timeout=60, # Increased from default ~30
    ping_interval=25 # Default is 25
    # No need for large max_http_buffer_size as files aren't sent via SocketIO
)

# --- Ensure directories exist ---
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"✔️ Directories verified/created.")
except OSError as e:
    print(f"❌ CRITICAL ERROR creating directories: {e}")
    sys.exit(1)

# --- Font Setup ---
# (Keep setup_font function as before)
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
# (Keep Constants as before)
TEXT_COLOR = (0, 0, 0); SHADOW_COLOR = (255, 255, 255); SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'


# --- Helper Functions ---
# (Keep emit_progress and emit_error as before)
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)
def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)


# --- Core Functions ---
# (Keep get_roboflow_predictions, extract_translation, ask_luminai,
# find_optimal_text_settings_final, draw_text_on_layer as before)
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=60): # Increased timeout slightly
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

def extract_translation(text):
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL);
    if match: return match.group(1).strip()
    else: return text.strip().strip('"\'').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    print("ℹ️ Calling LuminAI...")
    url = "https://ai.siputzx.my.id/"; payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}; timeout_seconds = 60 # Increased timeout slightly
    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            if response.status_code == 200:
                result_text = response.json().get("result", ""); translation = extract_translation(result_text.strip())
                print(f"✔️ LuminAI translation received: '{translation[:50]}...'")
                return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5)); print(f"⚠️ Rate limit (429). Retrying after {retry_after}s...")
                if sid: emit_progress(-1, f"Translation busy. Retrying...", -1, sid); socketio.sleep(retry_after)
            else:
                print(f"❌ LuminAI failed: Status {response.status_code} - {response.text[:150]}")
                if sid: emit_error(f"Translation failed (Status {response.status_code}).", sid)
                return "" # Stop retrying other errors
        except RequestException as e:
            print(f"❌ Network error LuminAI (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                print("   ❌ LuminAI failed after max retries (network).")
                if sid: emit_error("Translation connection failed.", sid)
                return ""
            else:
                wait_time = 2 * (attempt + 1); print(f"      Retrying in {wait_time}s..."); socketio.sleep(wait_time)
        except Exception as e:
            print(f"❌ Unexpected error LuminAI (Attempt {attempt+1}): {e}"); traceback.print_exc(limit=1)
            if attempt == max_retries - 1:
                 print("   ❌ LuminAI failed after max retries (unexpected).")
                 if sid: emit_error("Unexpected translation error.", sid); return ""
            else:
                 socketio.sleep(2)
    print("   ❌ LuminAI failed after all retries.");
    if sid: emit_error("Translation unavailable.", sid); return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    # (Keep implementation as before)
    print("ℹ️ Finding optimal text settings...")
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid: print("   ⚠️ Invalid polygon."); return None
    if not text: print("   ⚠️ Empty text."); return None
    if isinstance(text_formatter, DummyTextFormatter): print("   ⚠️ Cannot find settings: DummyFormatter active."); return None

    best_fit = None
    for font_size in range(65, 4, -1):
        font = text_formatter.get_font(font_size)
        if font is None: continue
        padding_distance = max(1.5, font_size * 0.12); text_fitting_polygon = None
        try:
            temp_poly = initial_shrunk_polygon.buffer(-padding_distance, join_style=2)
            if temp_poly.is_valid and not temp_poly.is_empty and isinstance(temp_poly, Polygon): text_fitting_polygon = temp_poly
            else:
                 temp_poly_fallback = initial_shrunk_polygon.buffer(-2.0, join_style=2)
                 if temp_poly_fallback.is_valid and not temp_poly_fallback.is_empty and isinstance(temp_poly_fallback, Polygon): text_fitting_polygon = temp_poly_fallback
                 else: continue
        except Exception as buffer_err: print(f"   ⚠️ Error buffering polygon size {font_size}: {buffer_err}. Skipping."); continue
        if not text_fitting_polygon: continue
        minx, miny, maxx, maxy = text_fitting_polygon.bounds; target_width = maxx - minx; target_height = maxy - miny
        if target_width <= 5 or target_height <= 10: continue
        wrapped_text = None
        try: wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        except Exception as layout_err: print(f"   ⚠️ Error in layout size {font_size}: {layout_err}. Skipping."); continue
        if not wrapped_text: continue
        try:
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_actual_width = m_bbox[2] - m_bbox[0]; text_actual_height = m_bbox[3] - m_bbox[1]
            shadow_offset = max(1, font_size // 18)
            if (text_actual_height + shadow_offset) <= target_height and (text_actual_width + shadow_offset) <= target_width:
                x_center_offset = (target_width - text_actual_width) / 2; y_center_offset = (target_height - text_actual_height) / 2
                draw_x = minx + x_center_offset - m_bbox[0]; draw_y = miny + y_center_offset - m_bbox[1]
                best_fit = {'text': wrapped_text, 'font': font, 'x': int(round(draw_x)), 'y': int(round(draw_y)), 'font_size': font_size};
                print(f"   ✔️ Optimal fit found: Size={font_size}, Pos=({best_fit['x']},{best_fit['y']})"); break
        except Exception as measure_err: print(f"   ⚠️ Error measuring size {font_size}: {measure_err}. Skipping."); continue
    if best_fit is None: print(f"   ⚠️ Warning: Could not find suitable font size for text: '{text[:30]}...'")
    return best_fit

def draw_text_on_layer(text_settings, image_size):
    # (Keep implementation as before)
    print("ℹ️ Drawing text layer...")
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
    if not text_settings or not isinstance(text_settings, dict): print("   ⚠️ Invalid text_settings."); return text_layer
    try:
        draw_on_layer = ImageDraw.Draw(text_layer); font = text_settings.get('font'); text_to_draw = text_settings.get('text', '');
        x = text_settings.get('x', 0); y = text_settings.get('y', 0); font_size = text_settings.get('font_size', 10)
        if not font or not text_to_draw: print("   ⚠️ Missing font or text."); return text_layer
        shadow_offset = max(1, font_size // 18); shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)
        draw_on_layer.multiline_text((x + shadow_offset, y + shadow_offset), text_to_draw, font=font, fill=shadow_color_with_alpha, align='center', spacing=4)
        draw_on_layer.multiline_text((x, y), text_to_draw, font=font, fill=TEXT_COLOR + (255,), align='center', spacing=4)
        print(f"   ✔️ Drew text '{text_to_draw[:20]}...' at ({x},{y}) size {font_size}"); return text_layer
    except Exception as e: print(f"❌ Error in draw_text_on_layer: {e}"); traceback.print_exc(limit=1); return Image.new('RGBA', image_size, (0, 0, 0, 0))

# --- Main Processing Task ---
# (Keep process_image_task core logic largely as before, it already works with a path)
# It now assumes the file at image_path already exists.
def process_image_task(image_path, output_filename_base, mode, sid):
    """ Core logic: Processes image found at image_path. """
    print(f"ℹ️ SID {sid}: Starting image processing task for {image_path}")
    start_time = time.time(); inpainted_image_cv = None; final_image_np = None; translations_list = []
    final_output_path = ""; result_data = {}; image = None
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    ROBOFLOW_BUBBLE_DETECT_URL = 'https://outline.roboflow.com/yolo-0kqkh/2'
    try:
        # === Step 0: Load Image (from existing path) ===
        emit_progress(0, "Loading image...", 5, sid);
        # File existence is already checked in handle_start_processing before calling this
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path} (Task Level).")

        # (Rest of image loading/validation logic is the same)
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        result_image = image.copy(); text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # === Step 1: Remove Text ===
        emit_progress(1, "Detecting text...", 10, sid);
        retval, buffer_text = cv2.imencode('.jpg', image); # Encode original image for detection
        if not retval or buffer_text is None: raise ValueError("Failed encode text detect.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8'); text_predictions = []
        try:
            # (Text detection and inpainting logic remains the same)
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
                 print(f"   Inpainting detected text areas...")
                 inpainted_image_cv = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS) # Inpaint original
                 if inpainted_image_cv is None: raise RuntimeError("cv2.inpaint returned None")
                 result_image = inpainted_image_cv # Update result_image
                 print(f"   Inpainting complete.")
            else: print(f"   No text detected, skipping inpainting.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ SID {sid}: Error during Roboflow text detection: {rf_err}. Skipping text removal."); emit_error("Text detection failed.", sid)
        except Exception as e: print(f"❌ SID {sid}: Error during text detection/inpainting: {e}. Skipping text removal."); emit_error("Text processing error.", sid)


        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting bubbles...", 30, sid);
        # Use the same encoded image (b64_image_text) for bubble detection
        b64_bubble = b64_image_text
        # No need to re-encode, just check if it exists from Step 1
        if not b64_bubble:
             # This case should ideally not happen if Step 1 ran, but handle defensively
             print(f"⚠️ SID {sid}: Missing encoded image data for bubble detection. Trying to re-encode.")
             retval_b, buffer_b = cv2.imencode('.jpg', image)
             if retval_b and buffer_b is not None:
                 b64_bubble = base64.b64encode(buffer_b).decode('utf-8')
             else:
                raise ValueError("Missing base64 data & failed re-encode for bubble detection.")

        bubble_predictions = []
        try:
             # (Bubble detection logic remains the same)
            print(f"   Sending request to Roboflow bubble detection...")
            bubble_predictions = get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_bubble)
            print(f"   Found {len(bubble_predictions)} speech bubbles.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ SID {sid}: Error during Roboflow bubble detection: {rf_err}."); emit_error("Bubble detection failed.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        except Exception as e: print(f"❌ SID {sid}: Error during Roboflow bubble detection: {e}."); emit_error("Bubble detection error.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}


        if not bubble_predictions and not final_output_path:
             print(f"   SID {sid}: No speech bubbles detected.")
             # (Logic for no bubbles remains the same)
             emit_progress(4, "No bubbles detected.", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}


        # --- Bubble Processing Loop ---
        elif bubble_predictions and not final_output_path:
             # (Bubble processing loop logic remains exactly the same)
             emit_progress(2, f"Processing {len(bubble_predictions)} bubbles...", 40, sid)
             image_pil = None
             try: image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert('RGBA') # Convert cleaned image
             except Exception as pil_conv_err: print(f"❌ SID {sid}: Error converting to PIL: {pil_conv_err}"); emit_error("Cannot draw (PIL error).", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_pil_error.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

             if image_pil: # Only proceed if PIL conversion worked
                 temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1))); image_size = image_pil.size
                 bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90
                 original_image_for_cropping = image.copy() # Keep a copy of the *original* for cropping context

                 for i, pred in enumerate(bubble_predictions):
                     try: # Original style try/except per bubble
                          current_bubble_progress = base_progress + int(((i + 1) / bubble_count) * (max_progress_bubbles - base_progress))
                          emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                          points = pred.get("points", []);
                          if len(points) < 3: print("   Skipping bubble: Not enough points."); continue
                          coords = [(int(p["x"]), int(p["y"])) for p in points];

                          # === ORIGINAL SCRIPT GEOMETRY VALIDATION ===
                          original_polygon = Polygon(coords)
                          if not original_polygon.is_valid:
                              print("   Original polygon is invalid, attempting to fix...")
                              try: original_polygon = make_valid(original_polygon)
                              except Exception as mv_err: print(f"   Skipping bubble: make_valid failed: {mv_err}"); continue
                              if original_polygon.geom_type == 'MultiPolygon':
                                  print("   Fixed polygon resulted in MultiPolygon, selecting largest.")
                                  original_polygon = max(original_polygon.geoms, key=lambda p: p.area, default=None)
                              if not isinstance(original_polygon, Polygon) or original_polygon.is_empty or not original_polygon.is_valid:
                                  print("   Skipping bubble: Polygon could not be validated or became empty.")
                                  continue
                          # === END ORIGINAL GEOMETRY VALIDATION ===

                          print(f"   Bubble polygon area: {original_polygon.area:.1f} pixels")
                          minx_orig, miny_orig, maxx_orig, maxy_orig = map(int, original_polygon.bounds)
                          minx_orig, miny_orig = max(0, minx_orig), max(0, miny_orig); maxx_orig, maxy_orig = min(w_img, maxx_orig), min(h_img, maxy_orig)
                          if maxx_orig <= minx_orig or maxy_orig <= miny_orig: print("   Skipping bubble: Invalid crop."); continue

                          bubble_crop = original_image_for_cropping[miny_orig:maxy_orig, minx_orig:maxx_orig]
                          if bubble_crop.size == 0: print("   Skipping bubble: Crop empty."); continue
                          _, crop_buffer = cv2.imencode('.jpg', bubble_crop);
                          if crop_buffer is None: print("   Skipping bubble: Failed encode crop."); continue
                          crop_bytes = crop_buffer.tobytes()

                          print("   Requesting translation from LuminAI...")
                          translation_prompt_orig = 'ترجم نص المانجا هذا إلى اللغة العربية بحيث تكون الترجمة مفهومة وتوصل المعنى الى القارئ. أرجو إرجاع الترجمة فقط بين علامتي اقتباس مثل "النص المترجم". مع مراعاة النبرة والانفعالات الظاهرة في كل سطر (مثل: الصراخ، التردد، الهمس) وأن تُترجم بطريقة تُحافظ على الإيقاع المناسب للفقاعة.'
                          translation = ask_luminai(translation_prompt_orig, crop_bytes, sid=sid) # Use original prompt
                          if not translation: print("   Skipping bubble: Translation failed or empty."); continue # Skip if empty like original
                          print(f"   Translation received: '{translation}'")

                          if mode == 'extract': translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                          elif mode == 'auto':
                              width_orig = maxx_orig - minx_orig; height_orig = maxy_orig - miny_orig; initial_buffer_distance = max(3.0, (width_orig + height_orig) / 2 * 0.10)
                              initial_shrunk_polygon = None
                              try: # Original shrink logic
                                   shrunk = original_polygon.buffer(-initial_buffer_distance, join_style=2)
                                   if not shrunk.is_valid or shrunk.is_empty: shrunk_fallback = original_polygon.buffer(-3.0, join_style=2); initial_shrunk_polygon = shrunk_fallback
                                   else: initial_shrunk_polygon = shrunk
                                   if not initial_shrunk_polygon.is_valid or initial_shrunk_polygon.is_empty or initial_shrunk_polygon.geom_type != 'Polygon': print("   Warning: Shrunk polygon invalid. Using original."); initial_shrunk_polygon = original_polygon
                              except Exception as initial_buffer_err: print(f"   Warning: Error shrinking: {initial_buffer_err}. Using original."); initial_shrunk_polygon = original_polygon
                              if not isinstance(initial_shrunk_polygon, Polygon) or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid: print("   Skipping bubble: Final polygon invalid."); continue

                              arabic_text = text_formatter.format_arabic_text(translation) # Direct call
                              if not arabic_text: print("   Skipping bubble: Formatted text empty."); continue

                              print("   Finding optimal font size and layout...")
                              text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, initial_shrunk_polygon)
                              if text_settings:
                                   print(f"   Optimal settings found: Size {text_settings['font_size']}, Pos ({text_settings['x']}, {text_settings['y']})")
                                   print("   Drawing text layer...")
                                   text_layer = draw_text_on_layer(text_settings, image_size)
                                   if text_layer: print("   Compositing text layer..."); image_pil.paste(text_layer, (0, 0), text_layer); processed_count += 1
                                   else: print("   Skipping bubble: draw_text_on_layer failed.")
                              else: print("   Skipping bubble: Could not fit text.")
                     except Exception as bubble_proc_err: print(f"❌ Error processing bubble {i + 1}: {bubble_proc_err}"); traceback.print_exc(); emit_progress(3, f"Skipping bubble {i+1} (error).", current_bubble_progress, sid); continue

                 # --- Finalize after loop ---
                 # (Finalization logic remains the same)
                 if mode == 'extract': emit_progress(4, f"Finished extracting ({processed_count}/{bubble_count}).", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
                 elif mode == 'auto':
                     emit_progress(4, f"Finished drawing ({processed_count}/{bubble_count}).", 95, sid)
                     try: final_image_rgb = image_pil.convert('RGB'); final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR); output_filename = f"{output_filename_base}_translated.jpg"; result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
                     except Exception as convert_err: print(f"❌ SID {sid}: Error converting final PIL: {convert_err}. Saving cleaned."); emit_error("Failed finalize translated.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_err.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
                 if not final_output_path: final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)


        # === Step 5 & 6: Save and Complete ===
        if final_image_np is not None and final_output_path:
            # (Saving logic remains the same)
            emit_progress(5, "Saving final image...", 98, sid); save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np);
                 if not save_success: raise IOError(f"cv2.imwrite failed: {final_output_path}")
                 print(f"✔️ Saved (OpenCV): {final_output_path}")
            except Exception as cv_save_err:
                 print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL...");
                 try: pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB)); os.makedirs(os.path.dirname(final_output_path), exist_ok=True); pil_img_to_save.save(final_output_path); save_success = True; print(f"✔️ Saved (PIL): {final_output_path}")
                 except Exception as pil_save_err: print(f"❌ PIL save failed: {pil_save_err}"); emit_error("Failed save final image.", sid)

            if save_success:
                processing_time = time.time() - start_time; print(f"✔️ SID {sid} Complete {processing_time:.2f}s."); emit_progress(6, f"Complete ({processing_time:.2f}s).", 100, sid)
                # --- ENSURE RESULT DATA IS SET ---
                if not result_data: # Should be set above, but fallback just in case
                    print(f"⚠️ SID {sid}: Result data was empty before completion. Creating default based on mode.")
                    result_data = {
                        'mode': mode,
                        'imageUrl': f'/results/{os.path.basename(final_output_path)}'
                    }
                    if mode == 'extract':
                        result_data['translations'] = translations_list
                # --- END ENSURE ---
                socketio.emit('processing_complete', result_data, room=sid)
            else: print(f"❌❌❌ SID {sid}: Critical Error: Could not save image {final_output_path}")
        elif not final_output_path: print(f"❌ SID {sid}: Aborted before output path set.")
        else: print(f"❌ SID {sid}: No final image data."); emit_error("Internal error: No final image.", sid)

    except Exception as e:
        print(f"❌❌❌ SID {sid}: UNHANDLED FATAL ERROR in task: {e}")
        traceback.print_exc()
        emit_error(f"Unexpected server error during processing ({type(e).__name__}).", sid)
    finally:
        # Clean up the originally uploaded file
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 SID {sid}: Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            print(f"⚠️ SID {sid}: Error cleaning up {image_path}: {cleanup_err}")


# --- Flask Routes ---

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve result images
@app.route('/results/<path:filename>')
def get_result_image(filename):
    # Basic security check
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    results_dir = os.path.abspath(app.config['RESULT_FOLDER'])
    try:
        return send_from_directory(results_dir, filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

# === NEW: HTTP Route for File Upload ===
@app.route('/upload', methods=['POST'])
def handle_upload():
    # Note: We don't have SID here easily. Errors are returned as JSON.
    # Client-side JS needs to handle these JSON errors.
    temp_log_id = f"upload_{uuid.uuid4()}" # ID for logging this specific upload attempt
    print(f"--- [{temp_log_id}] Received POST upload request ---")

    if 'file' not in request.files:
        print(f"[{temp_log_id}] ❌ Upload Error: No file part")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        print(f"[{temp_log_id}] ❌ Upload Error: No selected file")
        return jsonify({'error': 'No selected file'}), 400

    # Validate extension using secure_filename and ALLOWED_EXTENSIONS
    filename = secure_filename(file.filename) # Sanitize filename
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            print(f"[{temp_log_id}] ❌ Upload Error: Invalid file type '{ext}'")
            return jsonify({'error': f'Invalid file type: {ext}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    else:
         print(f"[{temp_log_id}] ❌ Upload Error: File has no extension")
         return jsonify({'error': 'File has no extension'}), 400

    # Generate unique filename structure
    unique_id = uuid.uuid4()
    # Use the validated & secured extension
    input_filename = f"{unique_id}.{ext}"
    output_filename_base = f"{unique_id}" # Base name for output files
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

    try:
        # Ensure upload folder exists (might be redundant but safe)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Save the file - Flask/Werkzeug handles streaming efficiently
        file.save(upload_path)
        file_size_kb = os.path.getsize(upload_path) / 1024
        print(f"[{temp_log_id}] ✔️ File saved via POST: {upload_path} ({file_size_kb:.1f} KB)")

        # Return details needed by the client to trigger processing via SocketIO
        return jsonify({
            'message': 'File uploaded successfully',
            'output_filename_base': output_filename_base,
            'saved_filename': input_filename # Send back the exact saved name
        }), 200
    except IOError as io_err:
        print(f"[{temp_log_id}] ❌ Upload Error: File write error: {io_err}")
        # Clean up partial file if it exists
        if os.path.exists(upload_path):
            try: os.remove(upload_path); print(f"[{temp_log_id}] 🧹 Cleaned up partial file after IO error.")
            except Exception: pass
        return jsonify({'error': 'Server error saving file'}), 500
    except Exception as e:
        # Catch potential errors during save (e.g., disk full, permissions)
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

# === MODIFIED: SocketIO Handler to Trigger Processing ===
@socketio.on('start_processing')
def handle_start_processing(data):
    """
    Receives filename info (after successful POST upload) and mode,
    then starts the background processing task.
    """
    sid = request.sid
    print(f"\n--- Received 'start_processing' event SID: {sid} ---")
    if not isinstance(data, dict):
        emit_error("Invalid request data.", sid); return

    # Expect 'output_filename_base' AND 'saved_filename' from client, plus 'mode'
    output_filename_base = data.get('output_filename_base')
    saved_filename = data.get('saved_filename') # The exact filename saved by /upload
    mode = data.get('mode')

    print(f"   Data received: {data}") # Log received keys/values

    # Validate required data
    if not output_filename_base:
        emit_error('Missing file identifier (output_filename_base).', sid); return
    if not saved_filename:
        emit_error('Missing saved filename.', sid); return
    if mode not in ['extract', 'auto']:
        emit_error(f"Invalid mode ('{mode}').", sid); return

    print(f"   Mode: '{mode}'")
    upload_dir = app.config['UPLOAD_FOLDER']

    # Construct the full path to the *already uploaded* file
    # Use secure_filename again on the received filename as an extra precaution
    # Although it should already be secured from the /upload response
    upload_path = os.path.join(upload_dir, secure_filename(saved_filename))

    # --- CRITICAL CHECK ---
    # Verify the file actually exists at the constructed path before starting task
    if not os.path.exists(upload_path) or not os.path.isfile(upload_path):
        print(f"❌ ERROR SID {sid}: File not found at path deduced from event: {upload_path}")
        # Use the filename the client *thinks* it uploaded in the error message
        emit_error(f"Uploaded file '{saved_filename}' not found on server. Please try uploading again.", sid)
        return
    # --- END CRITICAL CHECK ---

    # Optional: You could re-verify size here if needed, but MAX_CONTENT_LENGTH handled it earlier
    # file_size = os.path.getsize(upload_path)
    # if file_size == 0:
    #     emit_error("Uploaded file appears to be empty.", sid)
    #     return

    print(f"   File confirmed exists: {upload_path}")
    print(f"   Attempting to start background task...")

    try:
        # Start the background task, passing the path to the existing file
        socketio.start_background_task(
            process_image_task,
            image_path=upload_path, # Pass the full path
            output_filename_base=output_filename_base, # Pass the base name for naming results
            mode=mode,
            sid=sid
        )
        print(f"   ✔️ Task initiated SID {sid} for {saved_filename}.")
        # Notify client that processing has started
        socketio.emit('processing_started', {'message': 'File received, processing started...'}, room=sid)
    except Exception as task_start_err:
        print(f"❌ CRITICAL SID {sid}: Failed to start background task: {task_start_err}")
        traceback.print_exc()
        emit_error(f"Server error starting image processing task.", sid)
        # Note: We don't delete the uploaded file here, as the user might want to retry

    print(f"--- Finished handle 'start_processing' SID: {sid} ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App (Flask + SocketIO + Eventlet) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default Flask SECRET_KEY!")

    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Ext: {ALLOWED_EXTENSIONS}")
    print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async: {socketio.async_mode}")
    print(f"   * Roboflow Key: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Upload Limit: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f} MB")
    print(f"   * SocketIO Ping Timeout: {socketio.ping_timeout}s")

    print(f"   * Starting server http://0.0.0.0:{port}")
    print("   * NOTE: For production, consider using Gunicorn with eventlet workers or Waitress.")

    try:
        # Use socketio.run for development with eventlet support
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
        # For production with Gunicorn:
        # gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 your_app_file:app
        # For production with Waitress:
        # waitress-serve --host 0.0.0.0 --port 5000 your_app_file:app
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}")
        sys.exit(1)

