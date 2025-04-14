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
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
import uuid
# Make sure jsonify is imported
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback

eventlet.monkey_patch()

# --- Define DummyTextFormatter Globally First ---
class DummyTextFormatter:
    def __init__(self): print("⚠️ WARNING: Initializing DummyTextFormatter.")
    def set_arabic_font_path(self, path): pass
    def get_font(self, size): return None
    def format_arabic_text(self, text): return text
    def layout_balanced_text(self, draw, text, font, target_width): return text

# --- IMPORT YOUR MODULE ---
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
UPLOAD_FOLDER = 'uploads'; RESULT_FOLDER = 'results'; ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
# --- Flask App Setup ---
app = Flask(__name__); app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER; app.config['RESULT_FOLDER'] = RESULT_FOLDER;
# This now correctly limits HTTP upload size
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB Limit

# REMOVED max_http_buffer_size - Not needed for upload part anymore
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", logger=False, engineio_logger=False)

# --- Ensure directories exist ---
try: os.makedirs(UPLOAD_FOLDER, exist_ok=True); os.makedirs(RESULT_FOLDER, exist_ok=True); print(f"✔️ Directories verified/created.")
except OSError as e: print(f"❌ CRITICAL ERROR creating directories: {e}"); sys.exit(1)

# --- Font Setup ---
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
TEXT_COLOR = (0, 0, 0); SHADOW_COLOR = (255, 255, 255); SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    # Use socketio.emit correctly outside of request context
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01)
def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    # Use socketio.emit correctly outside of request context
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01)

# --- Core Functions (get_roboflow_predictions, extract_translation, ask_luminai, find_optimal_text_settings_final, draw_text_on_layer) ---
# These functions remain largely the same as the last correct version using original logic
# Make sure ask_luminai uses socketio.sleep
def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=30):
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown Model"; print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key: raise ValueError("Missing Roboflow API Key.")
    try:
        response = requests.post(f"{endpoint_url}?api_key={api_key}", data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status(); data = response.json(); predictions = data.get("predictions", []); print(f"✔️ Roboflow ({model_name}) response received. Predictions: {len(predictions)}")
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
    url = "https://luminai.my.id/"; payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}
    headers = {"Content-Type": "application/json", "Accept-Language": "ar"}; timeout_seconds = 45
    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            if response.status_code == 200:
                result_text = response.json().get("result", ""); translation = extract_translation(result_text.strip())
                print(f"✔️ LuminAI translation received: '{translation[:50]}...'"); return translation
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 5)); print(f"⚠️ Rate limit (429). Retrying after {retry_after}s...")
                if sid: emit_progress(-1, f"Translation busy. Retrying...", -1, sid); socketio.sleep(retry_after) # Use SocketIO sleep
            else:
                print(f"❌ LuminAI failed: Status {response.status_code} - {response.text[:150]}")
                if sid: emit_error(f"Translation failed (Status {response.status_code}).", sid); return ""
        except RequestException as e:
            print(f"❌ Network error LuminAI (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1: print("   ❌ LuminAI failed after max retries (network)."); emit_error("Translation connection failed.", sid); return ""
            else: wait_time = 2 * (attempt + 1); print(f"      Retrying in {wait_time}s..."); socketio.sleep(wait_time) # Use SocketIO sleep
        except Exception as e:
            print(f"❌ Unexpected error LuminAI (Attempt {attempt+1}): {e}"); traceback.print_exc(limit=1)
            if attempt == max_retries - 1: print("   ❌ LuminAI failed after max retries (unexpected)."); emit_error("Unexpected translation error.", sid); return ""
            else: socketio.sleep(2) # Use SocketIO sleep
    print("   ❌ LuminAI failed after all retries."); emit_error("Translation unavailable.", sid); return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    print("ℹ️ Finding optimal text settings...")
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid: print("   ⚠️ Invalid polygon."); return None
    if not text: print("   ⚠️ Empty text."); return None
    if isinstance(text_formatter, DummyTextFormatter): print("   ⚠️ Cannot find settings: DummyFormatter active."); return None

    best_fit = None;
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


# --- Main Processing Task (Takes filepath, uses Original Logic) ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """ Core logic: Uses original script's geometry validation & text handling. """
    start_time = time.time(); inpainted_image_cv = None; final_image_np = None; translations_list = []
    final_output_path = ""; result_data = {}; image = None
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    ROBOFLOW_BUBBLE_DETECT_URL = 'https://outline.roboflow.com/yolo-0kqkh/2'
    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid); image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Could not load image: {image_path}.")
        if len(image.shape) == 2 or image.shape[2] == 1: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError(f"Unsupported channels: {image.shape[2]}.")
        h_img, w_img = image.shape[:2];
        if h_img == 0 or w_img == 0: raise ValueError("Image zero dimensions.")
        result_image = image.copy(); text_mask = np.zeros(image.shape[:2], dtype=np.uint8) # Init like original

        # === Step 1: Remove Text ===
        emit_progress(1, "Detecting text...", 10, sid);
        retval, buffer_text = cv2.imencode('.jpg', image); # Use original 'image'
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
                 print(f"   Inpainting detected text areas...")
                 inpainted_image_cv = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS) # Inpaint original
                 if inpainted_image_cv is None: raise RuntimeError("cv2.inpaint returned None")
                 result_image = inpainted_image_cv # Update result_image
                 print(f"   Inpainting complete.")
            else: print(f"   No text detected, skipping inpainting.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ Error during Roboflow text detection: {rf_err}. Skipping text removal."); emit_error("Text detection failed.", sid)
        except Exception as e: print(f"❌ Error during text detection/inpainting: {e}. Skipping text removal."); emit_error("Text processing error.", sid)

        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting bubbles...", 30, sid); b64_bubble = b64_image_text
        if not b64_bubble: raise ValueError("Missing base64 data for bubble detection.")
        bubble_predictions = []
        try:
            print(f"   Sending request to Roboflow bubble detection...")
            bubble_predictions = get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_bubble)
            print(f"   Found {len(bubble_predictions)} speech bubbles.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ Error during Roboflow bubble detection: {rf_err}."); emit_error("Bubble detection failed.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        except Exception as e: print(f"❌ Error during Roboflow bubble detection: {e}."); emit_error("Bubble detection error.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        if not bubble_predictions and not final_output_path:
             print("   No speech bubbles detected.")
             emit_progress(4, "No bubbles detected.", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        # --- Bubble Processing Loop (Original Logic) ---
        elif bubble_predictions and not final_output_path:
             emit_progress(2, f"Processing {len(bubble_predictions)} bubbles...", 40, sid)
             image_pil = None
             try: image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert('RGBA') # Convert cleaned image
             except Exception as pil_conv_err: print(f"❌ Error converting to PIL: {pil_conv_err}"); emit_error("Cannot draw (PIL error).", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_pil_error.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

             if image_pil: # Only proceed if PIL conversion worked
                 temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1))); image_size = image_pil.size
                 bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90
                 original_image_for_cropping = image.copy() # Keep original for cropping

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
                          bubble_crop = original_image_for_cropping[miny_orig:maxy_orig, minx_orig:maxx_orig] # Crop original copy
                          if bubble_crop.size == 0: print("   Skipping bubble: Crop empty."); continue
                          _, crop_buffer = cv2.imencode('.jpg', bubble_crop);
                          if crop_buffer is None: print("   Skipping bubble: Failed encode crop."); continue
                          crop_bytes = crop_buffer.tobytes()

                          print("   Requesting translation from LuminAI...")
                          translation_prompt_orig = 'ترجم نص المانجا هذا إلى اللغة العربية بحيث تكون الترجمة مفهومة وتوصل المعنى الى القارئ. أرجو إرجاع الترجمة فقط بين علامتي اقتباس مثل "النص المترجم". مع مراعاة النبرة والانفعالات الظاهرة في كل سطر (مثل: الصراخ، التردد، الهمس) وأن تُترجم بطريقة تُحافظ على الإيقاع المناسب للفقاعة.'
                          translation = ask_luminai(translation_prompt_orig, crop_bytes, sid=sid)
                          if not translation: print("   Skipping bubble: Translation failed or empty."); continue
                          print(f"   Translation received: '{translation}'")

                          if mode == 'extract': translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                          elif mode == 'auto':
                              # === NO Dummy Check ===
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
                 if mode == 'extract': emit_progress(4, f"Finished extracting ({processed_count}/{bubble_count}).", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
                 elif mode == 'auto':
                     emit_progress(4, f"Finished drawing ({processed_count}/{bubble_count}).", 95, sid)
                     try: final_image_rgb = image_pil.convert('RGB'); final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR); output_filename = f"{output_filename_base}_translated.jpg"; result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
                     except Exception as convert_err: print(f"❌ Error converting final PIL: {convert_err}. Saving cleaned."); emit_error("Failed finalize translated.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_err.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
                 if not final_output_path: final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

        # === Step 5 & 6: Save and Complete ===
        if final_image_np is not None and final_output_path:
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
                # --- CORRECTED RESULT DATA BLOCK ---
                if not result_data:
                    print("⚠️ Result data empty. Creating default.")
                    result_data = {
                        'mode': mode,
                        'imageUrl': f'/results/{os.path.basename(final_output_path)}'
                    }
                    if mode == 'extract':
                        result_data['translations'] = translations_list
                # --- END CORRECTION ---
                socketio.emit('processing_complete', result_data, room=sid)
            else: print(f"❌❌❌ Critical Error: Could not save image {sid}")
        elif not final_output_path: print(f"❌ SID {sid}: Aborted before output path set.")
        else: print(f"❌ SID {sid}: No final image data."); emit_error("Internal error: No final image.", sid)

    except Exception as e: print(f"❌❌❌ UNHANDLED FATAL ERROR task {sid}: {e}"); traceback.print_exc(); emit_error(f"Unexpected server error ({type(e).__name__}).", sid)
    finally:
        try:
            if image_path and os.path.exists(image_path): os.remove(image_path); print(f"🧹 Cleaned up: {image_path}")
        except Exception as cleanup_err: print(f"⚠️ Error cleaning up {image_path}: {cleanup_err}")


# ==================================================
# NEW UPLOAD ROUTE and MODIFIED SOCKETIO HANDLERS
# ==================================================

# --- NEW: Flask Route for HTTP File Upload ---
@app.route('/upload', methods=['POST'])
def handle_file_upload():
    # Get SocketIO SID and mode from query parameters
    # Client-side JS needs to add these when calling fetch('/upload?sid=...')
    sid = request.args.get('sid')
    mode = request.args.get('mode', 'auto') # Default to 'auto' if not provided

    if not sid:
        print("❌ Upload Error: Missing sid query parameter.")
        # Cannot easily emit error to specific client without sid
        return jsonify({"error": "Missing session identifier"}), 400

    print(f"\n--- Received HTTP Upload for SID: {sid}, Mode: {mode} ---")

    if 'file' not in request.files:
        print(f"   SID {sid}: Upload Error: No file part in request.")
        emit_error('Upload error: No file part.', sid)
        return jsonify({"error": "No file part"}), 400

    file = request.files['file'] # Get the file object

    if file.filename == '':
        print(f"   SID {sid}: Upload Error: No file selected.")
        emit_error('Upload error: No file selected.', sid)
        return jsonify({"error": "No selected file"}), 400

    # --- Basic File Validation ---
    filename = file.filename # Use werkzeug's secure_filename? Maybe not needed for internal processing
    print(f"   SID {sid}: Received file: {filename}")
    file_ext = ""
    if '.' in filename:
        file_ext = filename.rsplit('.', 1)[1].lower()

    if file_ext not in ALLOWED_EXTENSIONS:
         print(f"   SID {sid}: Upload Error: Invalid file type: {file_ext}")
         emit_error(f'Upload error: Invalid file type ({file_ext}).', sid)
         return jsonify({"error": f"Invalid file type: {file_ext}"}), 400

    # --- Save the File ---
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.isdir(upload_dir) or not os.access(upload_dir, os.W_OK):
        print(f"❌ ERROR: Upload directory '{upload_dir}' inaccessible.")
        emit_error("Server config error.", sid)
        return jsonify({"error": "Server configuration error"}), 500

    upload_path = None
    try:
        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_ext}"
        output_filename_base = f"{unique_id}"
        upload_path = os.path.join(upload_dir, input_filename)

        file.save(upload_path) # Use Flask's save method for uploaded files
        print(f"   SID {sid}: File saved via HTTP: {upload_path}")

        # --- Start Background Task ---
        print(f"   SID {sid}: Attempting to start background task...")
        try:
            socketio.start_background_task(
                process_image_task,
                upload_path,          # Pass the path to the saved file
                output_filename_base,
                mode,
                sid
            )
            print(f"   SID {sid}: Background task initiated successfully.")
            # Send confirmation *back* to the specific client via SocketIO
            emit_progress(0, 'Upload successful! Processing started...', 5, sid) # Reuse emit_progress
            # Return success response for the HTTP request
            return jsonify({"message": "Upload successful, processing started."}), 200

        except Exception as task_start_err:
            print(f"❌ CRITICAL: Failed start task for SID {sid}: {task_start_err}"); traceback.print_exc()
            emit_error(f"Server error starting task.", sid)
            # Cleanup the uploaded file if task fails to start
            if upload_path and os.path.exists(upload_path):
                 try: os.remove(upload_path); print(f"   🧹 Cleaned up '{upload_path}'")
                 except Exception as cl_err: print(f"   ⚠️ Error cleaning up: {cl_err}")
            return jsonify({"error": "Server failed to start processing"}), 500

    except Exception as e:
        # Catch errors during validation or saving
        print(f"❌ Error handling upload for SID {sid}: {e}")
        traceback.print_exc()
        emit_error(f'Server error during upload: {type(e).__name__}', sid)
        # Cleanup partially saved file if it exists
        if upload_path and os.path.exists(upload_path):
             try: os.remove(upload_path); print(f"   🧹 Cleaned up '{upload_path}'")
             except Exception as cl_err: print(f"   ⚠️ Error cleaning up: {cl_err}")
        return jsonify({"error": "Server error handling upload"}), 500

# --- MODIFIED: SocketIO Handlers ---
@socketio.on('connect')
def handle_connect():
    # Send the client their SID upon connection, they might need it for the upload URL
    print(f"✅ Client connected: {request.sid}")
    emit('your_sid', {'sid': request.sid}) # Send SID back to the connecting client

@socketio.on('disconnect')
def handle_disconnect():
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing_request') # Renamed event to avoid confusion
def handle_start_request(data):
    """
    Handles the INITIAL request from the client BEFORE file upload.
    It just acknowledges the request and tells the client to proceed with HTTP upload.
    """
    sid = request.sid
    mode = data.get('mode', 'auto') # Client should send the mode here
    print(f"\n--- Received 'start_processing_request' event SID: {sid} ---")
    print(f"   Mode selected: '{mode}'")

    # Tell the client to initiate the actual file upload via HTTP POST
    # The client needs to include its sid in the POST request URL
    emit('initiate_http_upload', {'mode': mode}, room=sid) # Send back mode for confirmation

    print(f"--- Instructed SID {sid} to initiate HTTP upload ---")


# --- Flask Routes (Results and Index) ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/results/<path:filename>')
def get_result_image(filename):
    if '..' in filename or filename.startswith('/'): return "Invalid filename", 400
    results_dir = os.path.abspath(app.config['RESULT_FOLDER'])
    try: return send_from_directory(results_dir, filename, as_attachment=False)
    except FileNotFoundError: return "File not found", 404

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App (HTTP Upload Mode) ---")
    if not ROBOFLOW_API_KEY: print("███ WARNING: ROBOFLOW_API_KEY env var not set! ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production': print("⚠️ WARNING: Using default Flask SECRET_KEY!")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App: {app.name}"); print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}"); print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Ext: {ALLOWED_EXTENSIONS}"); print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async: {socketio.async_mode}"); print(f"   * Roboflow Key: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Max Upload Size (HTTP): {app.config['MAX_CONTENT_LENGTH'] // 1024 // 1024} MB")
    print(f"   * Starting server http://0.0.0.0:{port}")
    try: socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err: print(f"❌❌❌ Failed start server: {run_err}"); sys.exit(1)

