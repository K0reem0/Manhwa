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
import uuid
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback

eventlet.monkey_patch()

# --- Define DummyTextFormatter Globally First ---
class DummyTextFormatter:
    """A fallback class used when the real text_formatter module fails to import."""
    def __init__(self):
        print("⚠️ WARNING: Initializing DummyTextFormatter.")
        # Store a flag to easily identify this object later if needed
        self.is_dummy = True
    def set_arabic_font_path(self, path):
        pass # No-op
    def get_font(self, size):
        # Cannot reliably return a font, so return None
        print(f"   DummyFormatter: get_font({size}) -> None")
        return None
    def format_arabic_text(self, text):
        # Returning raw text is safest if real formatting unavailable.
        print("   DummyFormatter: format_arabic_text -> returning raw text")
        return text
    def layout_balanced_text(self, draw, text, font, target_width):
        # Cannot perform layout, return raw text
        print("   DummyFormatter: layout_balanced_text -> returning raw text")
        return text

# --- IMPORT YOUR MODULE ---
try:
    import text_formatter # Assuming text_formatter.py is in the same directory
    print("✔️ Successfully imported 'text_formatter.py'.")
    # Check if required functions exist
    if not all(hasattr(text_formatter, func) for func in ['set_arabic_font_path', 'get_font', 'format_arabic_text', 'layout_balanced_text']):
         print("⚠️ WARNING: 'text_formatter.py' seems to be missing required functions!")
         raise ImportError("Missing functions in text_formatter")
    # Optional: Add flag for consistency, useful if you change the check later
    # setattr(text_formatter, 'is_dummy', False)
except ImportError as import_err:
    print(f"❌ ERROR: Cannot import 'text_formatter.py' or it's incomplete: {import_err}")
    # Import failed, instantiate the globally defined DummyTextFormatter
    text_formatter = DummyTextFormatter() # Assign the dummy instance to the global variable
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
    sys.exit(1) # Exit if directories cannot be created

# --- Font Setup ---
def setup_font():
    font_path_to_set = None
    try:
        # Try relative path first (useful for Docker/deployments)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path_rel = os.path.join(script_dir, "fonts", "66Hayah.otf")
        # Try path relative to current working directory (useful for local dev)
        potential_path_cwd = os.path.join(".", "fonts", "66Hayah.otf")

        if os.path.exists(potential_path_rel):
            font_path_to_set = potential_path_rel
        elif os.path.exists(potential_path_cwd):
             font_path_to_set = potential_path_cwd

        if font_path_to_set:
            print(f"ℹ️ Font found: '{font_path_to_set}'. Setting path.")
            # Only call set_arabic_font_path if text_formatter is NOT the dummy
            # Check using the class name which is now guaranteed to exist
            if not isinstance(text_formatter, DummyTextFormatter):
                 text_formatter.set_arabic_font_path(font_path_to_set)
            else:
                 print("   (Dummy formatter active, skipping font path set)")

        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found in expected locations. Using default.")
            if not isinstance(text_formatter, DummyTextFormatter):
                text_formatter.set_arabic_font_path(None)
            else:
                 print("   (Dummy formatter active, font path remains unset)")

    except Exception as e:
        print(f"❌ Error during font path finding: {e}. Using default.")
        try:
             if not isinstance(text_formatter, DummyTextFormatter):
                text_formatter.set_arabic_font_path(None)
        except Exception as E2:
            print(f"❌ Error setting font path to None: {E2}")
setup_font()

# --- Constants ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    """ Sends progress updates to the client via SocketIO. """
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01) # Allow eventlet context switching

def emit_error(message, sid):
    """ Sends error messages to the client via SocketIO. """
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01) # Allow eventlet context switching

# --- Integrated Core Logic Functions ---
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=30):
    """Calls a Roboflow inference endpoint and returns predictions."""
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
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        predictions = data.get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as timeout_err:
         print(f"❌ Roboflow ({model_name}) Timeout Error: {timeout_err}")
         raise ConnectionError(f"Roboflow API ({model_name}) timed out after {timeout}s.") from timeout_err
    except requests.exceptions.HTTPError as http_err:
         print(f"❌ Roboflow ({model_name}) HTTP Error: Status {http_err.response.status_code}")
         try:
             error_detail = http_err.response.text[:200]
             print(f"   Response text: {error_detail}")
         except Exception: pass
         raise ConnectionError(f"Roboflow API ({model_name}) request failed (Status {http_err.response.status_code}). Check Key/URL.") from http_err
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Roboflow ({model_name}) Request Error: {req_err}")
        raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e:
        print(f"❌ Roboflow ({model_name}) Unexpected Error: {e}")
        traceback.print_exc(limit=2)
        raise RuntimeError(f"Unexpected error during Roboflow ({model_name}) request.") from e

def extract_translation(text):
    """Extracts text within the first pair of double quotes."""
    if not isinstance(text, str):
        return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip().strip('"\'').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    """Sends request to LuminAI, handles retries, and extracts translation."""
    print("ℹ️ Calling LuminAI...")
    url = "https://luminai.my.id/" # Make sure this is the correct endpoint
    payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}
    timeout_seconds = 45 # Increased timeout

    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)

            if response.status_code == 200:
                result_text = response.json().get("result", "")
                translation = extract_translation(result_text)
                if translation:
                    print(f"✔️ LuminAI translation received: '{translation[:50]}...'")
                    return translation
                else:
                    print(f"   ⚠️ LuminAI returned success but no text extracted from: '{result_text[:100]}...'")
                    return ""
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5))
                print(f"   ⚠️ LuminAI Rate limit (429). Retrying in {retry_after}s...")
                if sid: emit_progress(-1, f"Translation service busy. Retrying ({attempt+1}/{max_retries})...", -1, sid)
                socketio.sleep(retry_after)
            else:
                print(f"   ❌ LuminAI request failed: Status {response.status_code} - {response.text[:150]}")
                if sid: emit_error(f"Translation service failed (Status {response.status_code}).", sid)
                return ""

        except RequestException as e:
            print(f"   ❌ Network/Timeout error during LuminAI (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1)
                print(f"      Retrying in {wait_time}s...")
                socketio.sleep(wait_time)
            else:
                print("   ❌ LuminAI failed after max retries due to network issues.")
                if sid: emit_error("Translation service connection failed.", sid)
                return ""
        except Exception as e:
            print(f"   ❌ Unexpected error during LuminAI (Attempt {attempt+1}): {e}")
            traceback.print_exc(limit=1)
            if attempt < max_retries - 1:
                socketio.sleep(2)
            else:
                print("   ❌ LuminAI failed after max retries due to unexpected error.")
                if sid: emit_error("Unexpected error during translation.", sid)
                return ""

    print("   ❌ LuminAI failed after all retries.")
    if sid: emit_error("Translation service unavailable after multiple retries.", sid)
    return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """
    Searches for the best font size and text layout to fit within the polygon.
    Uses text_formatter.layout_balanced_text.
    """
    print("ℹ️ Finding optimal text settings...")
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid:
        print("   ⚠️ Invalid/empty polygon passed to find_optimal_text_settings.")
        return None
    if not text:
        print("   ⚠️ Empty text passed to find_optimal_text_settings.")
        return None
    # Check if the formatter object is the dummy instance
    if isinstance(text_formatter, DummyTextFormatter):
        print("   ⚠️ Cannot find optimal settings: DummyTextFormatter is active.")
        return None

    best_fit = None
    for font_size in range(65, 4, -1):
        # Use the globally available text_formatter object (real or dummy)
        font = text_formatter.get_font(font_size)
        if font is None:
            # This can happen if the real get_font fails or if using the dummy
            # print(f"   Debug: Failed to get font for size {font_size}")
            continue

        padding_distance = max(1.5, font_size * 0.12)
        text_fitting_polygon = None
        try:
            temp_poly = initial_shrunk_polygon.buffer(-padding_distance, join_style=2)
            if temp_poly.is_valid and not temp_poly.is_empty and isinstance(temp_poly, Polygon):
                text_fitting_polygon = temp_poly
            else:
                 temp_poly_fallback = initial_shrunk_polygon.buffer(-2.0, join_style=2)
                 if temp_poly_fallback.is_valid and not temp_poly_fallback.is_empty and isinstance(temp_poly_fallback, Polygon):
                     text_fitting_polygon = temp_poly_fallback
                 else:
                     continue
        except Exception as buffer_err:
            print(f"   ⚠️ Error buffering polygon for font size {font_size}: {buffer_err}. Skipping size.")
            continue

        if not text_fitting_polygon: continue

        minx, miny, maxx, maxy = text_fitting_polygon.bounds
        target_width = maxx - minx
        target_height = maxy - miny

        if target_width <= 5 or target_height <= 10:
            continue

        wrapped_text = None
        try:
             # Use the globally available text_formatter object (real or dummy)
             wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        except Exception as layout_err:
             print(f"   ⚠️ Error in layout_balanced_text for font size {font_size}: {layout_err}. Skipping size.")
             continue

        if not wrapped_text:
            continue

        try:
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_actual_width = m_bbox[2] - m_bbox[0]
            text_actual_height = m_bbox[3] - m_bbox[1]
            shadow_offset = max(1, font_size // 15)

            if (text_actual_height + shadow_offset) <= target_height and \
               (text_actual_width + shadow_offset) <= target_width:
                x_center_offset = (target_width - text_actual_width) / 2
                y_center_offset = (target_height - text_actual_height) / 2
                draw_x = minx + x_center_offset - m_bbox[0]
                draw_y = miny + y_center_offset - m_bbox[1]

                best_fit = {
                    'text': wrapped_text,
                    'font': font,
                    'x': int(round(draw_x)),
                    'y': int(round(draw_y)),
                    'font_size': font_size
                }
                print(f"   ✔️ Optimal fit found: Size={font_size}, Pos=({best_fit['x']},{best_fit['y']})")
                break

        except Exception as measure_err:
            print(f"   ⚠️ Error measuring text dimensions for size {font_size}: {measure_err}. Skipping size.")
            continue

    if best_fit is None:
        print(f"   ⚠️ Warning: Could not find suitable font size for text: '{text[:30]}...' within the given polygon.")

    return best_fit

def draw_text_on_layer(text_settings, image_size):
    """Draws the text with shadow onto a new transparent layer."""
    print("ℹ️ Drawing text layer...")
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))

    if not text_settings or not isinstance(text_settings, dict):
         print("   ⚠️ Invalid or missing text_settings passed to draw_text_on_layer.")
         return text_layer

    try:
        draw_on_layer = ImageDraw.Draw(text_layer)
        font = text_settings.get('font')
        text_to_draw = text_settings.get('text', '')
        x = text_settings.get('x', 0)
        y = text_settings.get('y', 0)
        font_size = text_settings.get('font_size', 10)

        if not font or not text_to_draw:
             print("   ⚠️ Missing font or text in text_settings for drawing.")
             return text_layer

        # Use the globally available text_formatter object if needed for font details,
        # but the font object itself should be passed in text_settings
        if isinstance(text_formatter, DummyTextFormatter) and font is None:
             print("   ⚠️ Cannot draw text: Dummy formatter active and no font provided.")
             return text_layer

        shadow_offset = max(1, font_size // 18)
        shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)

        draw_on_layer.multiline_text(
            (x + shadow_offset, y + shadow_offset), text_to_draw, font=font,
            fill=shadow_color_with_alpha, align='center', spacing=4
        )
        draw_on_layer.multiline_text(
            (x, y), text_to_draw, font=font,
            fill=TEXT_COLOR + (255,), align='center', spacing=4
        )
        print(f"   ✔️ Drew text '{text_to_draw[:20]}...' at ({x},{y}) with size {font_size}")
        return text_layer

    except Exception as e:
        print(f"❌ Error in draw_text_on_layer: {e}")
        traceback.print_exc(limit=1)
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
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    ROBOFLOW_BUBBLE_DETECT_URL = 'https://outline.roboflow.com/yolo-0kqkh/2'

    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image at {image_path}.")
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported image format (channels: {image.shape[2]}).")
        h_img, w_img = image.shape[:2]
        if h_img == 0 or w_img == 0: raise ValueError("Image loaded with zero width or height.")
        original_image_for_cropping = image.copy()
        result_image = image.copy()

        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        retval, buffer_text = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not retval or buffer_text is None: raise ValueError("Failed to encode image for text detection.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8')

        text_predictions = []
        try:
            text_predictions = get_roboflow_predictions(ROBOFLOW_TEXT_DETECT_URL, ROBOFLOW_API_KEY, b64_image_text)
        except (ValueError, ConnectionError, RuntimeError) as rf_err:
            print(f"⚠️ Roboflow text detection failed: {rf_err}. Proceeding without inpainting.")
            emit_progress(1, f"Text detection failed ({type(rf_err).__name__}), skipping removal.", 15, sid)
        except Exception as generic_err:
             print(f"⚠️ Unexpected error during text detection: {generic_err}. Proceeding without inpainting.")
             traceback.print_exc(limit=1)
             emit_progress(1, f"Text detection error, skipping removal.", 15, sid)

        emit_progress(1, f"Masking Text (Found {len(text_predictions)} areas)...", 15, sid)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons_drawn = 0
        if text_predictions:
            for pred in text_predictions:
                 points = pred.get("points", [])
                 if len(points) >= 3:
                     try:
                        polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                        polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1); polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                        cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                     except KeyError as ke: print(f"⚠️ Warn: Malformed point data in text prediction: {ke}")
                     except Exception as fill_err: print(f"⚠️ Warn: Error drawing text polygon: {fill_err}")

        if np.any(text_mask) and polygons_drawn > 0:
            emit_progress(1, f"Inpainting {polygons_drawn} text area(s)...", 20, sid)
            try:
                 inpainted_image = cv2.inpaint(result_image, text_mask, 10, cv2.INPAINT_NS)
                 if inpainted_image is None: raise RuntimeError("cv2.inpaint returned None")
                 emit_progress(1, "Inpainting complete.", 25, sid)
            except Exception as inpaint_err:
                 print(f"❌ Error during inpainting: {inpaint_err}. Using original image.")
                 emit_error(f"Inpainting failed: {inpaint_err}", sid); inpainted_image = result_image.copy()
        else:
            emit_progress(1, "No text found or masked to remove.", 25, sid); inpainted_image = result_image.copy()

        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        retval_bubble, buffer_bubble = cv2.imencode('.jpg', inpainted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not retval_bubble or buffer_bubble is None: raise ValueError("Failed to encode inpainted image for bubble detection.")
        b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8')

        bubble_predictions = []
        try:
             bubble_predictions = get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_bubble)
        except (ValueError, ConnectionError, RuntimeError) as rf_err:
             print(f"❌ Bubble detection failed: {rf_err}. Cannot proceed.")
             emit_error(f"Bubble detection failed ({type(rf_err).__name__}).", sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        except Exception as generic_err:
             print(f"❌ Unexpected error during bubble detection: {generic_err}. Cannot proceed.")
             traceback.print_exc(limit=1)
             emit_error(f"Bubble detection error ({type(generic_err).__name__}).", sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        if not bubble_predictions and not final_output_path:
             emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        elif bubble_predictions and not final_output_path:
             emit_progress(2, f"Bubble Detection Found: {len(bubble_predictions)} bubbles.", 40, sid)

             # === Step 3 & 4: Process Bubbles & Finalize ===
             image_pil = None; temp_draw_for_settings = None; image_size = (w_img, h_img)

             if mode == 'auto':
                 try:
                      image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                      temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
                 except Exception as pil_conv_err:
                      print(f"❌ Error converting image to PIL: {pil_conv_err}. Falling back to 'extract' mode.")
                      emit_error(f"Cannot draw text (Image conversion failed).", sid); mode = 'extract'

             bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90

             for i, pred in enumerate(bubble_predictions):
                 current_bubble_progress = base_progress + int(((i + 1) / bubble_count) * (max_progress_bubbles - base_progress))
                 emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)

                 try:
                      points = pred.get("points", [])
                      if len(points) < 3: print(f"   ⚠️ Skipping bubble {i+1}: Not enough points."); continue
                      coords = [(int(p["x"]), int(p["y"])) for p in points]
                      bubble_polygon = Polygon(coords)

                      # --- More Robust Geometry Validation ---
                      if not bubble_polygon.is_valid:
                          #print(f"   Debug: Bubble {i+1} polygon initially invalid. Attempting make_valid.")
                          original_coords_str = str(coords) # For logging if needed
                          try: fixed_geom = make_valid(bubble_polygon)
                          except Exception as geom_fix_err: print(f"   ⚠️ Skipping bubble {i+1}: Error during make_valid: {geom_fix_err}"); continue
                          #print(f"   Debug: make_valid result type: {fixed_geom.geom_type}")

                          if isinstance(fixed_geom, Polygon) and not fixed_geom.is_empty:
                              bubble_polygon = fixed_geom
                              #print(f"   Debug: Bubble {i+1} fixed to Polygon.")
                          elif isinstance(fixed_geom, MultiPolygon) and not fixed_geom.is_empty:
                              #print(f"   Debug: Bubble {i+1} fixed to MultiPolygon. Taking largest.")
                              largest_poly = max(fixed_geom.geoms, key=lambda p: p.area, default=None)
                              if isinstance(largest_poly, Polygon) and not largest_poly.is_empty: bubble_polygon = largest_poly
                              else: print(f"   ⚠️ Skipping bubble {i+1}: No valid Polygon in MultiPolygon after make_valid."); continue
                          else:
                              print(f"   ⚠️ Skipping bubble {i+1}: Invalid geometry type ({fixed_geom.geom_type}) after make_valid.")
                              # print(f"      Original Coords: {original_coords_str[:100]}...") # Optional debug log
                              continue
                      elif bubble_polygon.is_empty:
                           print(f"   ⚠️ Skipping bubble {i+1}: Initial polygon is empty."); continue
                      # --- End Geometry Validation ---

                      minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                      padding = 5
                      minx_c=max(0,minx-padding); miny_c=max(0,miny-padding); maxx_c=min(w_img,maxx+padding); maxy_c=min(h_img,maxy+padding)
                      if maxx_c<=minx_c or maxy_c<=miny_c: print(f"   ⚠️ Skipping bubble {i+1}: Invalid crop dimensions."); continue

                      bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                      if bubble_crop.size == 0: print(f"   ⚠️ Skipping bubble {i+1}: Crop empty."); continue
                      retval_crop, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop)
                      if not retval_crop or crop_buffer_enc is None: print(f"   ⚠️ Skipping bubble {i+1}: Failed encode crop."); continue
                      crop_bytes = crop_buffer_enc.tobytes()

                      translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, max_retries=2, sid=sid)
                      if not translation:
                          print(f"   ⚠️ Skipping bubble {i+1}: Translation failed."); translation = "[ترجمة فشلت]"

                      if mode == 'extract':
                           translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                      elif mode == 'auto' and image_pil and temp_draw_for_settings:
                           if translation == "[ترجمة فشلت]": print(f"   ⚠️ Skipping drawing bbl {i+1}: Translation failed."); continue

                           # --- Check if using Dummy Formatter (SHOULD WORK NOW) ---
                           if isinstance(text_formatter, DummyTextFormatter):
                               print(f"   ⚠️ Skipping drawing for bubble {i+1}: Dummy Text Formatter active.")
                               continue
                           # --- End Check ---

                           arabic_text = text_formatter.format_arabic_text(translation)
                           if not arabic_text: print(f"   ⚠️ Skipping bubble {i+1}: Formatted text empty."); continue

                           poly_width = maxx - minx; poly_height = maxy - miny
                           initial_buffer_distance = np.clip((poly_width + poly_height) / 2 * 0.10, 3.0, 15.0)
                           text_poly = None
                           try:
                                shrunk = bubble_polygon.buffer(-initial_buffer_distance, join_style=2)
                                if shrunk.is_valid and not shrunk.is_empty and isinstance(shrunk, Polygon): text_poly = shrunk
                                else:
                                     shrunk_fallback = bubble_polygon.buffer(-3.0, join_style=2)
                                     if shrunk_fallback.is_valid and not shrunk_fallback.is_empty and isinstance(shrunk_fallback, Polygon): text_poly = shrunk_fallback
                           except Exception as buffer_err: print(f"   ⚠️ Warn: buffer error {buffer_err}, using original poly.")
                           if not text_poly: text_poly = bubble_polygon # Fallback if buffering failed

                           if not text_poly or text_poly.is_empty: print(f"   ⚠️ Skipping bubble {i+1}: No valid text polygon."); continue

                           text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                           if text_settings:
                                text_layer = draw_text_on_layer(text_settings, image_size)
                                if text_layer:
                                     try: image_pil.paste(text_layer, (0, 0), text_layer); processed_count += 1; print(f"   ✔️ Pasted text layer bbl {i+1}")
                                     except Exception as paste_err: print(f"❌ Paste Error bbl {i+1}: {paste_err}")
                                else: print(f"⚠️ Draw func failed bbl {i+1}")
                           else: print(f"⚠️ Text fit failed bbl {i+1}")

                 except Exception as bubble_err:
                      print(f"❌❌❌ Unhandled error processing bubble {i + 1}: {bubble_err}")
                      traceback.print_exc(limit=2)
                      emit_progress(3, f"Skipping bubble {i+1} (error).", current_bubble_progress, sid)

             if mode == 'extract':
                 emit_progress(4, f"Finished extracting ({processed_count}/{bubble_count}).", 95, sid)
                 final_image_np = inpainted_image; output_filename = f"{output_filename_base}_cleaned.jpg"
                 result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
             elif mode == 'auto':
                 emit_progress(4, f"Finished drawing ({processed_count}/{bubble_count}).", 95, sid)
                 if image_pil:
                      try:
                           final_image_np = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                           output_filename = f"{output_filename_base}_translated.jpg"
                           result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
                      except Exception as convert_err:
                           print(f"❌ Error converting final PIL->CV2: {convert_err}. Saving cleaned.")
                           emit_error("Failed to finalize translated image.", sid)
                           final_image_np = inpainted_image
                           output_filename = f"{output_filename_base}_cleaned_conversion_error.jpg"
                           result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
                 else:
                      print("⚠️ 'auto' mode finished but PIL image unavailable. Saving cleaned.")
                      final_image_np = inpainted_image
                      output_filename = f"{output_filename_base}_cleaned_pil_error.jpg"
                      result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

             if not final_output_path:
                  final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        # === Step 5: Save Final Image ===
        if final_image_np is not None and final_output_path:
            emit_progress(5, "Saving final image...", 98, sid)
            save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np)
                 if not save_success: raise IOError(f"cv2.imwrite returned False: {final_output_path}")
                 print(f"✔️ Saved final image (OpenCV): {final_output_path}")
            except Exception as cv_save_err:
                 print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL...")
                 try:
                     pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                     os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                     pil_img_to_save.save(final_output_path)
                     save_success = True
                     print(f"✔️ Saved final image (PIL): {final_output_path}")
                 except Exception as pil_save_err:
                      print(f"❌ PIL save also failed: {pil_save_err}")
                      emit_error("Failed to save final image.", sid)

            # === Step 6: Signal Completion ===
            if save_success:
                processing_time = time.time() - start_time
                print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Output: {final_output_path}")
                emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
                if not result_data:
                    print("⚠️ Result data empty before emit. Creating default.")
                    result_data = {'mode': mode, 'imageUrl': f'/results/{os.path.basename(final_output_path)}'}
                    if mode == 'extract': result_data['translations'] = translations_list
                socketio.emit('processing_complete', result_data, room=sid)
            else:
                print(f"❌❌❌ Critical Error: Could not save image for SID {sid}")

        elif not final_output_path: print(f"❌ SID {sid}: Processing aborted before output path set.")
        else: print(f"❌ SID {sid}: No final image data to save."); emit_error("Internal error: No final image generated.", sid)

    except Exception as e:
        print(f"❌❌❌ UNHANDLED FATAL ERROR in process_image_task for SID {sid}: {e}")
        traceback.print_exc()
        emit_error(f"Unexpected server error ({type(e).__name__}). Check logs.", sid)

    finally:
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path); print(f"🧹 Cleaned up: {image_path}")
        except Exception as cleanup_err: print(f"⚠️ Error cleaning up {image_path}: {cleanup_err}")


# --- Flask Routes & SocketIO Handlers ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/results/<path:filename>')
def get_result_image(filename):
    if '..' in filename or filename.startswith('/'): return "Invalid filename", 400
    results_dir = os.path.abspath(app.config['RESULT_FOLDER'])
    try: return send_from_directory(results_dir, filename, as_attachment=False)
    except FileNotFoundError: return "File not found", 404

@socketio.on('connect')
def handle_connect(): print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect(): print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    sid = request.sid
    print(f"\n--- Received 'start_processing' event from SID: {sid} ---")
    if not isinstance(data, dict): emit_error("Invalid request format.", sid); return
    print(f"   Data keys received: {list(data.keys())}")
    file_data_str = data.get('file')
    mode = data.get('mode')
    if not file_data_str or not isinstance(file_data_str, str) or not file_data_str.startswith('data:image'): emit_error('Invalid/missing image data URI.', sid); return
    if mode not in ['extract', 'auto']: emit_error(f"Invalid/missing mode ('{mode}').", sid); return
    print(f"   Mode selected: '{mode}'")

    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(upload_dir) or not os.access(upload_dir, os.W_OK):
        print(f"❌ ERROR: Upload directory '{upload_dir}' not accessible.")
        emit_error("Server configuration error.", sid); return

    upload_path = None
    try:
        header, encoded = file_data_str.split(',', 1)
        file_extension_match = re.search(r'data:image/(\w+)', header)
        if not file_extension_match: emit_error('Could not determine file type.', sid); return
        file_extension = file_extension_match.group(1).lower()
        if file_extension not in ALLOWED_EXTENSIONS: emit_error(f'Invalid file type: {file_extension}.', sid); return

        file_bytes = base64.b64decode(encoded)
        if not file_bytes: emit_error('Empty file data.', sid); return
        if len(file_bytes) > app.config['MAX_CONTENT_LENGTH']: emit_error(f"File too large.", sid); return

        unique_id = uuid.uuid4(); input_filename = f"{unique_id}.{file_extension}"; output_filename_base = f"{unique_id}"
        upload_path = os.path.join(upload_dir, input_filename)
        with open(upload_path, 'wb') as f: f.write(file_bytes)
        print(f"   ✔️ File saved: {upload_path} ({len(file_bytes)/1024:.1f} KB)")

    except (ValueError, TypeError, IndexError) as decode_err: print(f"❌ Decode error: {decode_err}"); emit_error('Error processing image data.', sid); return
    except base64.binascii.Error as b64_err: print(f"❌ Base64 error: {b64_err}"); emit_error('Invalid Base64 data.', sid); return
    except IOError as io_err:
         print(f"❌ File write error: {io_err}"); emit_error('Server error saving upload.', sid)
         if upload_path and os.path.exists(upload_path):
             try: os.remove(upload_path); print(f"   🧹 Cleaned up partially saved file.")
             except Exception as cleanup_err_io: print(f"   ⚠️ Error cleaning up after IO error: {cleanup_err_io}")
         return
    except Exception as e:
        print(f"❌ Unexpected file handling error: {e}"); traceback.print_exc(); emit_error(f'Server upload error: {type(e).__name__}', sid)
        if upload_path and os.path.exists(upload_path):
             try: os.remove(upload_path); print(f"   🧹 Cleaned up file after unexpected save error.")
             except Exception as cleanup_err_unexp: print(f"   ⚠️ Error cleaning up after unexpected save error: {cleanup_err_unexp}")
        return

    if upload_path:
        print(f"   Attempting to start background processing task...")
        try:
            socketio.start_background_task( process_image_task, upload_path, output_filename_base, mode, sid )
            print(f"   ✔️ Background task for SID {sid} initiated."); socketio.emit('processing_started', {'message': 'Upload successful! Processing started...'}, room=sid)
        except Exception as task_start_err:
            print(f"❌ CRITICAL: Failed to start background task: {task_start_err}"); traceback.print_exc()
            emit_error(f"Server error starting task: {task_start_err}", sid)
            if os.path.exists(upload_path):
                try: os.remove(upload_path); print(f"   🧹 Cleaned up file due to task start failure.")
                except Exception as cleanup_err_start: print(f"   ⚠️ Error cleaning up after task start failure: {cleanup_err_start}")
    else: print("❌ Internal Error: upload_path not set."); emit_error("Internal server error.", sid)
    print(f"--- Finished handling 'start_processing' request for SID: {sid} ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default Flask SECRET_KEY!")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App Name: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Extensions: {ALLOWED_EXTENSIONS}")
    print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async Mode: {socketio.async_mode}")
    print(f"   * Roboflow Key Loaded: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Starting server on http://0.0.0.0:{port}")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err: print(f"❌❌❌ Failed to start SocketIO server: {run_err}"); sys.exit(1)

