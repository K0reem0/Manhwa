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
from flask import Flask, request, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
import eventlet
import traceback
from werkzeug.utils import secure_filename
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import io
import zipfile
import shutil

eventlet.monkey_patch()

# --- Define DummyTextFormatter Globally First ---
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
ALLOWED_ARCHIVES = {'zip'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

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
        # In a real app, you might want to create a default font if it's missing
        # if not font_path_to_set and os.path.exists("default_font.ttf"):
        #     text_formatter.set_arabic_font_path("default_font.ttf")
    except Exception as e:
        print(f"❌ Error finding font path: {e}. Using default.")
        try: text_formatter.set_arabic_font_path(None)
        except AttributeError: print("   (Formatter missing method? Skipping.)")
        except Exception as E2: print(f"❌ Error setting font path to None: {E2}")
setup_font()

# --- Constants ---
TEXT_COLOR = (0, 0, 0); SHADOW_COLOR = (255, 255, 255); SHADOW_OPACITY = 90
TRANSLATION_PROMPT_GEMINI = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'

# --- Initialize Google GenAI ---
model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)
        print("✔️ Google GenAI configured successfully.")
    except Exception as e:
        print(f"❌ ERROR: Failed to configure Google GenAI: {e}")
        print("   Please ensure GOOGLE_API_KEY is set correctly.")


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

def extract_translation(text):
    """Extracts the quoted translation from a Gemini response."""
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL);
    if match: return match.group(1).strip()
    else: return text.strip().strip('"\'').strip()

# --- MODIFIED: ask_gemini_translation function to use Google GenAI ---
def ask_gemini_translation(prompt, image_bytes, max_retries=3, sid=None):
    """Sends an image to Google Gemini for translation."""
    if not model:
        print("   ❌ Gemini model not initialized. Cannot perform translation.")
        if sid: emit_error("Translation service unavailable (Gemini not configured).", sid)
        return ""

    print("ℹ️ Calling Google Gemini for translation...")
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        gemini_image_data = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(io.BytesIO(image_bytes).getvalue()).decode('utf-8')
        }
    except Exception as e:
        print(f"❌ Error preparing image for Gemini: {e}")
        if sid: emit_error("Error processing image for translation.", sid)
        return ""

    contents = [
        {"text": prompt},
        {"inline_data": gemini_image_data}
    ]

    for attempt in range(max_retries):
        print(f"   Gemini Attempt {attempt + 1}/{max_retries}...")
        try:
            response = model.generate_content(contents)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"❌ Gemini response blocked: {response.prompt_feedback.block_reason}")
                if sid: emit_error(f"Translation blocked due to content policy.", sid)
                return ""
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"❌ Gemini returned no candidates or parts.")
                if sid: emit_error("Translation service returned empty result.", sid)
                return ""
            result_text = ""
            for part in response.candidates[0].content.parts:
                if part.text:
                    result_text += part.text
            translation = extract_translation(result_text.strip())
            print(f"✔️ Gemini translation received: '{translation[:50]}...'")
            return translation
        except Exception as e:
            print(f"❌ Error calling Gemini API (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                print("   ❌ Gemini failed after max retries.")
                if sid: emit_error("Translation service failed.", sid)
                return ""
            else:
                wait_time = 2 * (attempt + 1); print(f"      Retrying in {wait_time}s..."); socketio.sleep(wait_time)
    print("   ❌ Gemini failed after all retries.")
    if sid: emit_error("Translation unavailable.", sid); return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """Finds the best font size and position to fit text within a polygon."""
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
    """Draws text on a transparent layer with a shadow effect."""
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
def process_image_task(image_path, output_filename_base, mode, sid):
    """
    Main background task to process a single image.
    This function is called by both single image upload and batch processing.
    """
    print(f"ℹ️ SID {sid}: Starting image processing task for {os.path.basename(image_path)}")
    start_time = time.time(); inpainted_image_cv = None; final_image_np = None; translations_list = []
    final_output_path = ""; result_data = {}; image = None
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1'
    ROBOFLOW_BUBBLE_DETECT_URL = 'https://outline.roboflow.com/yolo-0kqkh/2'

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
                 print(f"   Inpainting detected text areas...")
                 inpainted_image_cv = cv2.inpaint(image, text_mask, 10, cv2.INPAINT_NS)
                 if inpainted_image_cv is None: raise RuntimeError("cv2.inpaint returned None")
                 result_image = inpainted_image_cv
                 print(f"   Inpainting complete.")
            else: print(f"   No text detected, skipping inpainting.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ SID {sid}: Error during Roboflow text detection: {rf_err}. Skipping text removal."); emit_error("Text detection failed.", sid)
        except Exception as e: print(f"❌ SID {sid}: Error during text detection/inpainting: {e}. Skipping text removal."); emit_error("Text processing error.", sid)

        emit_progress(2, "Detecting bubbles...", 30, sid);
        b64_bubble = b64_image_text
        if not b64_bubble:
             print(f"⚠️ SID {sid}: Missing encoded image data for bubble detection. Trying to re-encode.")
             retval_b, buffer_b = cv2.imencode('.jpg', image)
             if retval_b and buffer_b is not None:
                 b64_bubble = base64.b64encode(buffer_b).decode('utf-8')
             else:
                raise ValueError("Missing base64 data & failed re-encode for bubble detection.")

        bubble_predictions = []
        try:
            print(f"   Sending request to Roboflow bubble detection...")
            bubble_predictions = get_roboflow_predictions(ROBOFLOW_BUBBLE_DETECT_URL, ROBOFLOW_API_KEY, b64_bubble)
            print(f"   Found {len(bubble_predictions)} speech bubbles.")
        except (ValueError, ConnectionError, RuntimeError, requests.exceptions.RequestException) as rf_err: print(f"❌ SID {sid}: Error during Roboflow bubble detection: {rf_err}."); emit_error("Bubble detection failed.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        except Exception as e: print(f"❌ SID {sid}: Error during Roboflow bubble detection: {e}."); emit_error("Bubble detection error.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        if not bubble_predictions and not final_output_path:
             print(f"   SID {sid}: No speech bubbles detected.")
             emit_progress(4, "No bubbles detected.", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

        elif bubble_predictions and not final_output_path:
             emit_progress(2, f"Processing {len(bubble_predictions)} bubbles...", 40, sid)
             image_pil = None
             try: image_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
             except Exception as pil_conv_err: print(f"❌ SID {sid}: Error converting to PIL: {pil_conv_err}"); emit_error("Cannot draw (PIL error).", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_pil_error.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

             if image_pil:
                 temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1))); image_size = image_pil.size
                 bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90
                 original_image_for_cropping = image.copy()

                 for i, pred in enumerate(bubble_predictions):
                     try:
                          current_bubble_progress = base_progress + int(((i + 1) / bubble_count) * (max_progress_bubbles - base_progress))
                          emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                          points = pred.get("points", []);
                          if len(points) < 3: print("   Skipping bubble: Not enough points."); continue
                          coords = [(int(p["x"]), int(p["y"])) for p in points];

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
                          print(f"   Bubble polygon area: {original_polygon.area:.1f} pixels")
                          minx_orig, miny_orig, maxx_orig, maxy_orig = map(int, original_polygon.bounds)
                          minx_orig, miny_orig = max(0, minx_orig), max(0, miny_orig); maxx_orig, maxy_orig = min(w_img, maxx_orig), min(h_img, maxy_orig)
                          if maxx_orig <= minx_orig or maxy_orig <= miny_orig: print("   Skipping bubble: Invalid crop."); continue

                          bubble_crop = original_image_for_cropping[miny_orig:maxy_orig, minx_orig:maxx_orig]
                          if bubble_crop.size == 0: print("   Skipping bubble: Crop empty."); continue
                          _, crop_buffer = cv2.imencode('.jpg', bubble_crop);
                          if crop_buffer is None: print("   Skipping bubble: Failed encode crop."); continue
                          crop_bytes = crop_buffer.tobytes()

                          print("   Requesting translation from Gemini...")
                          translation = ask_gemini_translation(TRANSLATION_PROMPT_GEMINI, crop_bytes, sid=sid)
                          if not translation: print("   Skipping bubble: Translation failed or empty."); continue
                          print(f"   Translation received: '{translation}'")

                          if mode == 'extract': translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                          elif mode == 'auto':
                              width_orig = maxx_orig - minx_orig; height_orig = maxy_orig - miny_orig; initial_buffer_distance = max(3.0, (width_orig + height_orig) / 2 * 0.10)
                              initial_shrunk_polygon = None
                              try:
                                   shrunk = original_polygon.buffer(-initial_buffer_distance, join_style=2)
                                   if not shrunk.is_valid or shrunk.is_empty: shrunk_fallback = original_polygon.buffer(-3.0, join_style=2); initial_shrunk_polygon = shrunk_fallback
                                   else: initial_shrunk_polygon = shrunk
                                   if not isinstance(initial_shrunk_polygon, Polygon) or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid: print("   Warning: Shrunk polygon invalid. Using original."); initial_shrunk_polygon = original_polygon
                              except Exception as initial_buffer_err: print(f"   Warning: Error shrinking: {initial_buffer_err}. Using original."); initial_shrunk_polygon = original_polygon
                              if not isinstance(initial_shrunk_polygon, Polygon) or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid: print("   Skipping bubble: Final polygon invalid."); continue

                              arabic_text = text_formatter.format_arabic_text(translation)
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

                 if mode == 'extract': emit_progress(4, f"Finished extracting ({processed_count}/{bubble_count}).", 95, sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
                 elif mode == 'auto':
                     emit_progress(4, f"Finished drawing ({processed_count}/{bubble_count}).", 95, sid)
                     try: final_image_rgb = image_pil.convert('RGB'); final_image_np = cv2.cvtColor(np.array(final_image_rgb), cv2.COLOR_RGB2BGR); output_filename = f"{output_filename_base}_translated.jpg"; result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
                     except Exception as convert_err: print(f"❌ SID {sid}: Error converting final PIL: {convert_err}. Saving cleaned."); emit_error("Failed finalize translated.", sid); final_image_np = result_image; output_filename = f"{output_filename_base}_cleaned_err.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
                 if not final_output_path: final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)

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
                processing_time = time.time() - start_time;
                print(f"✔️ SID {sid} Complete {processing_time:.2f}s for {os.path.basename(image_path)}.");
                emit_progress(6, f"Complete ({processing_time:.2f}s).", 100, sid)
                if not result_data:
                    result_data = {'mode': mode, 'imageUrl': f'/results/{os.path.basename(final_output_path)}'}
                    if mode == 'extract': result_data['translations'] = translations_list
                # Add the original filename to the result data
                result_data['original_filename'] = os.path.basename(image_path)
                result_data['is_zip_batch'] = False # Default to False for single image
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

# --- Flask Routes ---
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

# --- NEW: Zip file upload route ---
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
    mode = data.get('mode')
    print(f"   Data received: {data}")
    if not output_filename_base or not saved_filename or mode not in ['extract', 'auto']:
        emit_error('Missing or invalid parameters.', sid); return
    print(f"   Mode: '{mode}'")
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
            mode=mode,
            sid=sid
        )
        print(f"   ✔️ Task initiated SID {sid} for {saved_filename}.")
        socketio.emit('processing_started', {'message': 'File received, processing started...'}, room=sid)
    except Exception as task_start_err:
        print(f"❌ CRITICAL SID {sid}: Failed to start background task: {task_start_err}")
        traceback.print_exc()
        emit_error(f"Server error starting image processing task.", sid)

# --- NEW: Batch processing handler for zip files ---
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
    if not isinstance(images_to_process, list) or not images_to_process or mode not in ['extract', 'auto']:
        emit_error('Invalid or empty image list for batch processing.', sid); return

    print(f"   Initiating batch processing for {len(images_to_process)} images.")
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
                mode=mode,
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
    if not GOOGLE_API_KEY: print("███ WARNING: GOOGLE_API_KEY env var not set! Gemini features will not work. ███")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Flask App: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Ext: {ALLOWED_EXTENSIONS}")
    print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async: {socketio.async_mode}")
    print(f"   * Roboflow Key: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Google API Key: {'Yes' if GOOGLE_API_KEY else 'NO (!)'}")
    print(f"   * Upload Limit: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f} MB")
    print(f"   * SocketIO Ping Timeout: {socketio.ping_timeout}s")
    print(f"   * Starting server http://0.0.0.0:{port}")
    print("   * NOTE: For production, consider using Gunicorn with eventlet workers or Waitress.")
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
        print(f"❌❌❌ Failed start server: {run_err}")
        sys.exit(1)
