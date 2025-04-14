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
            text_formatter.set_arabic_font_path(font_path_to_set)
        else:
            print("⚠️ Font 'fonts/66Hayah.otf' not found in expected locations. Using default.")
            text_formatter.set_arabic_font_path(None)
    except Exception as e:
        print(f"❌ Error during font path finding: {e}. Using default.")
        try:
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
    # print(f"SID: {sid} | Progress: Step {step}, {percentage}% - {message}") # Optional server-side logging
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid)
    socketio.sleep(0.01) # Allow eventlet context switching

def emit_error(message, sid):
    """ Sends error messages to the client via SocketIO. """
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}")
    socketio.emit('processing_error', {'error': message}, room=sid)
    socketio.sleep(0.01) # Allow eventlet context switching

# --- Integrated Core Logic Functions --- (Keep these as they were in the previous correct version)
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
             # Attempt to log part of the error response if available
             error_detail = http_err.response.text[:200]
             print(f"   Response text: {error_detail}")
         except Exception: pass
         raise ConnectionError(f"Roboflow API ({model_name}) request failed (Status {http_err.response.status_code}). Check Key/URL.") from http_err
    except requests.exceptions.RequestException as req_err:
        # Catch other request-related errors (DNS, connection refused, etc.)
        print(f"❌ Roboflow ({model_name}) Request Error: {req_err}")
        raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from req_err
    except Exception as e:
        # Catch unexpected errors during the request/response handling
        print(f"❌ Roboflow ({model_name}) Unexpected Error: {e}")
        traceback.print_exc(limit=2) # Print more traceback for unexpected errors
        raise RuntimeError(f"Unexpected error during Roboflow ({model_name}) request.") from e

def extract_translation(text):
    """Extracts text within the first pair of double quotes."""
    if not isinstance(text, str):
        return ""
    # Regex to find text inside the first pair of double quotes, handling potential whitespace
    match = re.search(r'"(.*?)"', text, re.DOTALL)
    if match:
        # Return the captured group, stripped of leading/trailing whitespace
        return match.group(1).strip()
    else:
        # Fallback: If no quotes found, maybe the API returned just the text.
        # Strip potential outer quotes (single or double) and whitespace just in case.
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
                translation = extract_translation(result_text) # Use helper here
                if translation:
                    print(f"✔️ LuminAI translation received: '{translation[:50]}...'")
                    return translation
                else:
                    print(f"   ⚠️ LuminAI returned success but no text extracted from: '{result_text[:100]}...'")
                    return "" # Return empty if extraction fails even on success
            elif response.status_code == 429:
                # Rate limited
                retry_after = int(response.headers.get('Retry-After', 5)) # Use Retry-After header if available
                print(f"   ⚠️ LuminAI Rate limit (429). Retrying in {retry_after}s...")
                if sid: emit_progress(-1, f"Translation service busy. Retrying ({attempt+1}/{max_retries})...", -1, sid) # Inform user
                socketio.sleep(retry_after) # Use socketio.sleep for async compatibility
            else:
                # Other HTTP errors (e.g., 500 Internal Server Error) - don't retry these by default
                print(f"   ❌ LuminAI request failed: Status {response.status_code} - {response.text[:150]}")
                if sid: emit_error(f"Translation service failed (Status {response.status_code}).", sid)
                return "" # Stop retrying on non-429 errors

        except RequestException as e:
            # Network errors (timeout, connection error, etc.)
            print(f"   ❌ Network/Timeout error during LuminAI (Attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 * (attempt + 1) # Exponential backoff
                print(f"      Retrying in {wait_time}s...")
                socketio.sleep(wait_time)
            else:
                print("   ❌ LuminAI failed after max retries due to network issues.")
                if sid: emit_error("Translation service connection failed.", sid)
                return "" # Failed after all retries
        except Exception as e:
            # Unexpected errors (e.g., JSON decoding error, etc.)
            print(f"   ❌ Unexpected error during LuminAI (Attempt {attempt+1}): {e}")
            traceback.print_exc(limit=1)
            if attempt < max_retries - 1:
                socketio.sleep(2) # Short wait before retry on unexpected error
            else:
                print("   ❌ LuminAI failed after max retries due to unexpected error.")
                if sid: emit_error("Unexpected error during translation.", sid)
                return "" # Failed after all retries

    # Should only be reached if all retries failed (e.g., due to 429s or network errors)
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
    if isinstance(text_formatter, DummyTextFormatter):
        print("   ⚠️ Cannot find optimal settings: DummyTextFormatter is active.")
        return None

    best_fit = None
    # Iterate font sizes from large to small
    for font_size in range(65, 4, -1):
        font = text_formatter.get_font(font_size)
        if font is None:
            # print(f"   Debug: Failed to get font for size {font_size}") # Enable for debugging font issues
            continue # Skip size if font loading fails

        # Calculate padding based on font size, ensuring a minimum
        padding_distance = max(1.5, font_size * 0.12)
        text_fitting_polygon = None
        try:
            # Shrink the initial polygon further specifically for text fitting
            temp_poly = initial_shrunk_polygon.buffer(-padding_distance, join_style=2) # MITRE join
            if temp_poly.is_valid and not temp_poly.is_empty and isinstance(temp_poly, Polygon):
                text_fitting_polygon = temp_poly
            else:
                 # Fallback buffer if the first attempt failed or resulted in non-polygon
                 temp_poly_fallback = initial_shrunk_polygon.buffer(-2.0, join_style=2)
                 if temp_poly_fallback.is_valid and not temp_poly_fallback.is_empty and isinstance(temp_poly_fallback, Polygon):
                     text_fitting_polygon = temp_poly_fallback
                 else:
                     # If fallback also fails, we can't proceed for this font size
                     continue

        except Exception as buffer_err:
            print(f"   ⚠️ Error buffering polygon for font size {font_size}: {buffer_err}. Skipping size.")
            continue

        # If after buffering (and fallback) we don't have a valid polygon, skip
        if not text_fitting_polygon: continue

        minx, miny, maxx, maxy = text_fitting_polygon.bounds
        target_width = maxx - minx
        target_height = maxy - miny

        # Basic sanity check on the target area size
        if target_width <= 5 or target_height <= 10:
            continue

        wrapped_text = None
        try:
             # --- Use text_formatter's layout function ---
             wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        except Exception as layout_err:
             print(f"   ⚠️ Error in layout_balanced_text for font size {font_size}: {layout_err}. Skipping size.")
             # traceback.print_exc(limit=1) # Enable for debugging layout errors
             continue # Skip if layout function fails

        if not wrapped_text:
            # print(f"   Debug: Layout returned empty for size {font_size}") # Debugging
            continue # Skip if layout returns empty

        try:
            # Measure the actual dimensions using Pillow's multiline_textbbox
            # Pillow's bbox is (left, top, right, bottom)
            # The temporary 'draw' context passed is sufficient for measurement
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')

            # Calculate width and height from the bounding box
            text_actual_width = m_bbox[2] - m_bbox[0]
            text_actual_height = m_bbox[3] - m_bbox[1]

            # Estimate needed space for shadow (adjust multiplier if needed)
            shadow_offset = max(1, font_size // 15) # Slightly larger shadow estimate

            # Check if the measured text dimensions (plus shadow estimate) fit within the target polygon's bounds
            if (text_actual_height + shadow_offset) <= target_height and \
               (text_actual_width + shadow_offset) <= target_width:
                # --- Fit found ---
                # Calculate position to center the text block within the target bounds
                x_center_offset = (target_width - text_actual_width) / 2
                y_center_offset = (target_height - text_actual_height) / 2

                # The draw position needs to be the top-left corner of the text block.
                # multiline_textbbox's top-left (m_bbox[0], m_bbox[1]) might not be (0,0)
                # depending on font metrics, especially for the first line.
                # We adjust the final draw position based on the bbox's origin and the target area's origin (minx, miny).
                draw_x = minx + x_center_offset - m_bbox[0]
                draw_y = miny + y_center_offset - m_bbox[1]

                best_fit = {
                    'text': wrapped_text,
                    'font': font,
                    'x': int(round(draw_x)), # Round to nearest pixel
                    'y': int(round(draw_y)), # Round to nearest pixel
                    'font_size': font_size
                }
                print(f"   ✔️ Optimal fit found: Size={font_size}, Pos=({best_fit['x']},{best_fit['y']})")
                break # Exit the loop once the largest possible font size fitting is found

        except Exception as measure_err:
            print(f"   ⚠️ Error measuring text dimensions for size {font_size}: {measure_err}. Skipping size.")
            # traceback.print_exc(limit=1) # Enable for debugging measurement errors
            continue

    if best_fit is None:
        print(f"   ⚠️ Warning: Could not find suitable font size for text: '{text[:30]}...' within the given polygon.")

    return best_fit

def draw_text_on_layer(text_settings, image_size):
    """Draws the text with shadow onto a new transparent layer."""
    print("ℹ️ Drawing text layer...")
    # Create a transparent layer matching the image size
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))

    if not text_settings or not isinstance(text_settings, dict):
         print("   ⚠️ Invalid or missing text_settings passed to draw_text_on_layer.")
         return text_layer # Return the empty layer

    try:
        draw_on_layer = ImageDraw.Draw(text_layer)

        # Safely get values from the dictionary
        font = text_settings.get('font')
        text_to_draw = text_settings.get('text', '')
        x = text_settings.get('x', 0)
        y = text_settings.get('y', 0)
        font_size = text_settings.get('font_size', 10) # Use a default if missing

        if not font or not text_to_draw:
             print("   ⚠️ Missing font or text in text_settings for drawing.")
             return text_layer # Return empty layer if essential info is missing

        # Calculate shadow offset based on font size
        shadow_offset = max(1, font_size // 18) # Adjust divisor for finer control
        # Combine shadow color with opacity
        shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)

        # Draw shadow slightly offset first
        draw_on_layer.multiline_text(
            (x + shadow_offset, y + shadow_offset),
            text_to_draw,
            font=font,
            fill=shadow_color_with_alpha,
            align='center',
            spacing=4 # Match spacing used in measurement
        )
        # Draw the main text on top
        draw_on_layer.multiline_text(
            (x, y),
            text_to_draw,
            font=font,
            fill=TEXT_COLOR + (255,), # Use TEXT_COLOR constant with full opacity
            align='center',
            spacing=4 # Match spacing used in measurement
        )
        print(f"   ✔️ Drew text '{text_to_draw[:20]}...' at ({x},{y}) with size {font_size}")
        return text_layer

    except Exception as e:
        print(f"❌ Error in draw_text_on_layer: {e}")
        traceback.print_exc(limit=1)
        # Return an empty layer on error to avoid crashing the composition step
        return Image.new('RGBA', image_size, (0, 0, 0, 0))

# --- Main Processing Task --- (Keep as previous correct version)
def process_image_task(image_path, output_filename_base, mode, sid):
    """ Core logic: Cleans, optionally translates, and draws text based on mode. """
    start_time = time.time()
    inpainted_image = None
    final_image_np = None
    translations_list = []
    final_output_path = ""
    result_data = {}
    # URLs for Roboflow Models (Consider making these constants or env vars)
    ROBOFLOW_TEXT_DETECT_URL = 'https://serverless.roboflow.com/text-detection-w0hkg/1' # Verify Model ID/Version
    ROBOFLOW_BUBBLE_DETECT_URL = 'https://outline.roboflow.com/yolo-0kqkh/2'       # Verify Model ID/Version

    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}. Check file.")
        # Ensure image is 3-channel BGR
        if len(image.shape) == 2: # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: # BGRA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: # Other (e.g., 1 channel but not grayscale?)
            raise ValueError(f"Unsupported image format (channels: {image.shape[2]}). Must be BGR, BGRA, or Grayscale.")
        h_img, w_img = image.shape[:2]
        if h_img == 0 or w_img == 0:
             raise ValueError("Image loaded with zero width or height.")
        original_image_for_cropping = image.copy() # Keep pristine original for accurate cropping
        result_image = image.copy() # This will be inpainted

        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        # Encode image for Roboflow API
        retval, buffer_text = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not retval or buffer_text is None:
            raise ValueError("Failed to encode image for text detection.")
        b64_image_text = base64.b64encode(buffer_text).decode('utf-8')

        text_predictions = []
        try:
            text_predictions = get_roboflow_predictions(
                ROBOFLOW_TEXT_DETECT_URL,
                ROBOFLOW_API_KEY, b64_image_text
            )
        except (ValueError, ConnectionError, RuntimeError) as rf_err:
            # Log Roboflow-specific errors but try to continue without inpainting
            print(f"⚠️ Roboflow text detection failed: {rf_err}. Proceeding without inpainting.")
            emit_progress(1, f"Text detection failed ({type(rf_err).__name__}), skipping removal.", 15, sid)
        except Exception as generic_err:
             # Catch any other unexpected errors during detection
             print(f"⚠️ Unexpected error during text detection: {generic_err}. Proceeding without inpainting.")
             traceback.print_exc(limit=1)
             emit_progress(1, f"Text detection error, skipping removal.", 15, sid)


        emit_progress(1, f"Masking Text (Found {len(text_predictions)} areas)...", 15, sid)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        polygons_drawn = 0
        if text_predictions: # Only create mask if detections exist
            for pred in text_predictions:
                 points = pred.get("points", [])
                 if len(points) >= 3: # Need at least 3 points for a polygon
                     try:
                        # Convert points, ensuring they are within image bounds
                        polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                        polygon_np[:, 0] = np.clip(polygon_np[:, 0], 0, w_img - 1)
                        polygon_np[:, 1] = np.clip(polygon_np[:, 1], 0, h_img - 1)
                        # Draw filled polygon onto the mask
                        cv2.fillPoly(text_mask, [polygon_np], 255)
                        polygons_drawn += 1
                     except KeyError as ke: print(f"⚠️ Warn: Malformed point data in text prediction: {ke}")
                     except Exception as fill_err: print(f"⚠️ Warn: Error drawing text polygon: {fill_err}")

        if np.any(text_mask) and polygons_drawn > 0:
            emit_progress(1, f"Inpainting {polygons_drawn} text area(s)...", 20, sid)
            try:
                 # Use INPAINT_NS (Navier-Stokes) - generally good for texture
                 # INPAINT_TELEA (Fast Marching Method) might be faster but sometimes leaves artifacts
                 inpainted_image = cv2.inpaint(result_image, text_mask, 10, cv2.INPAINT_NS) # Radius 10
                 if inpainted_image is None:
                     raise RuntimeError("cv2.inpaint returned None")
                 emit_progress(1, "Inpainting complete.", 25, sid)
            except Exception as inpaint_err:
                 print(f"❌ Error during inpainting: {inpaint_err}. Using original image without inpainting.")
                 emit_error(f"Inpainting failed: {inpaint_err}", sid)
                 inpainted_image = result_image.copy() # Fallback to non-inpainted image
        else:
            emit_progress(1, "No text found or masked to remove.", 25, sid)
            inpainted_image = result_image.copy() # No inpainting needed/done

        # --- At this point, `inpainted_image` holds the cleaned image (or original if cleaning failed/skipped) ---

        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        # Encode the (potentially inpainted) image for bubble detection
        retval_bubble, buffer_bubble = cv2.imencode('.jpg', inpainted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not retval_bubble or buffer_bubble is None:
            raise ValueError("Failed to encode inpainted image for bubble detection.")
        b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8')

        bubble_predictions = []
        try:
             bubble_predictions = get_roboflow_predictions(
                  ROBOFLOW_BUBBLE_DETECT_URL,
                  ROBOFLOW_API_KEY, b64_bubble
             )
        except (ValueError, ConnectionError, RuntimeError) as rf_err:
             # Bubble detection is critical for translation/drawing. If it fails, we stop here.
             print(f"❌ Bubble detection failed: {rf_err}. Cannot proceed with translation/drawing.")
             emit_error(f"Bubble detection failed ({type(rf_err).__name__}).", sid)
             # Save the cleaned image as the final result in this case
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
             # Go directly to saving (handled outside this try block by the presence of final_image_np and path)
             # We should not continue processing bubbles if detection fails.
        except Exception as generic_err:
             # Catch any other unexpected errors during bubble detection
             print(f"❌ Unexpected error during bubble detection: {generic_err}. Cannot proceed.")
             traceback.print_exc(limit=1)
             emit_error(f"Bubble detection error ({type(generic_err).__name__}).", sid)
             # Save cleaned image as final result
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
             # Go directly to saving


        # If bubble detection succeeded but found 0 bubbles
        if not bubble_predictions and not final_output_path: # Check final_output_path to ensure no error occurred above
             emit_progress(4, "No speech bubbles detected. Finishing.", 95, sid)
             final_image_np = inpainted_image
             output_filename = f"{output_filename_base}_cleaned.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
             # Go to saving step

        # Only proceed if bubble detection worked AND found bubbles AND no critical error occurred
        elif bubble_predictions and not final_output_path:
             emit_progress(2, f"Bubble Detection Found: {len(bubble_predictions)} bubbles.", 40, sid)

             # === Step 3 & 4: Process Bubbles (Translate/Draw) & Finalize ===
             image_pil = None
             temp_draw_for_settings = None
             image_size = (w_img, h_img) # Already have this from Step 0

             if mode == 'auto':
                 # Convert the inpainted CV2 image to PIL RGBA format for drawing text layers
                 try:
                      # Use the inpainted image for the base layer
                      image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA')
                      # Create a minimal dummy draw object for text measurement in find_optimal_text_settings
                      temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
                 except Exception as pil_conv_err:
                      print(f"❌ Error converting image to PIL format: {pil_conv_err}. Falling back to 'extract' mode.")
                      emit_error(f"Cannot draw text (Image conversion failed: {pil_conv_err}). Falling back to extract mode.", sid)
                      mode = 'extract' # Change mode if PIL conversion fails

             bubble_count = len(bubble_predictions)
             processed_count = 0
             base_progress = 45
             max_progress_bubbles = 90 # Allocate progress range for bubble processing

             for i, pred in enumerate(bubble_predictions):
                 current_bubble_progress = base_progress + int(((i + 1) / bubble_count) * (max_progress_bubbles - base_progress))
                 emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)

                 try:
                      # Extract and validate polygon points
                      points = pred.get("points", [])
                      if len(points) < 3:
                          print(f"   ⚠️ Skipping bubble {i+1}: Not enough points ({len(points)}).")
                          continue
                      coords = [(int(p["x"]), int(p["y"])) for p in points]
                      bubble_polygon = Polygon(coords)

                      # Validate and potentially fix the polygon geometry
                      if not bubble_polygon.is_valid:
                          bubble_polygon = make_valid(bubble_polygon)
                          # If fixing results in multiple polygons, take the largest one
                          if bubble_polygon.geom_type == 'MultiPolygon':
                              bubble_polygon = max(bubble_polygon.geoms, key=lambda p: p.area, default=None)
                          # If still not a valid Polygon, skip
                          if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty:
                              print(f"   ⚠️ Skipping bubble {i+1}: Invalid geometry after make_valid.")
                              continue
                      elif bubble_polygon.is_empty:
                           print(f"   ⚠️ Skipping bubble {i+1}: Empty polygon.")
                           continue


                      # --- Crop Original Image for Translation Context ---
                      # Use bounds of the valid polygon
                      minx, miny, maxx, maxy = map(int, bubble_polygon.bounds)
                      # Add a small padding around the bounds for context, clipped to image dimensions
                      padding = 5
                      minx_c = max(0, minx - padding)
                      miny_c = max(0, miny - padding)
                      maxx_c = min(w_img, maxx + padding)
                      maxy_c = min(h_img, maxy + padding)

                      # Ensure cropped dimensions are valid
                      if maxx_c <= minx_c or maxy_c <= miny_c:
                          print(f"   ⚠️ Skipping bubble {i+1}: Invalid crop dimensions after padding.")
                          continue

                      # Crop from the ORIGINAL image (before inpainting)
                      bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                      if bubble_crop.size == 0:
                           print(f"   ⚠️ Skipping bubble {i+1}: Crop resulted in empty image.")
                           continue

                      # Encode the cropped image for LuminAI
                      retval_crop, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop)
                      if not retval_crop or crop_buffer_enc is None:
                           print(f"   ⚠️ Skipping bubble {i+1}: Failed to encode crop.")
                           continue
                      crop_bytes = crop_buffer_enc.tobytes()

                      # --- Get Translation ---
                      translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, max_retries=2, sid=sid) # Reduce retries per bubble
                      if not translation:
                          print(f"   ⚠️ Skipping bubble {i+1}: Translation failed or returned empty.")
                          translation = "[ترجمة فشلت]" # Placeholder for extract mode if needed


                      # --- Mode Specific Actions ---
                      if mode == 'extract':
                           translations_list.append({'id': i + 1, 'translation': translation})
                           processed_count += 1
                      elif mode == 'auto' and image_pil and temp_draw_for_settings:
                           # Proceed only if translation was successful for drawing
                           if translation == "[ترجمة فشلت]" or not translation:
                               print(f"   ⚠️ Skipping drawing for bubble {i+1}: No valid translation.")
                               continue
                           if isinstance(text_formatter, DummyTextFormatter):
                               print(f"   ⚠️ Skipping drawing for bubble {i+1}: Text Formatter is not properly loaded.")
                               continue

                           # Format text using the imported module
                           arabic_text = text_formatter.format_arabic_text(translation)
                           if not arabic_text:
                               print(f"   ⚠️ Skipping bubble {i+1}: Formatted text is empty.")
                               continue

                           # --- Shrink Polygon for Text Fitting ---
                           # Calculate buffer distance based on polygon size (similar to original script)
                           poly_width = maxx - minx # Use original bounds before padding
                           poly_height = maxy - miny
                           # Dynamic buffer based on average dimension, with min/max caps
                           initial_buffer_distance = np.clip((poly_width + poly_height) / 2 * 0.10, 3.0, 15.0)
                           text_poly = None # Polygon to fit text into

                           try:
                                shrunk = bubble_polygon.buffer(-initial_buffer_distance, join_style=2) # MITRE join
                                # Check if shrinking worked and resulted in a valid Polygon
                                if shrunk.is_valid and not shrunk.is_empty and isinstance(shrunk, Polygon):
                                    text_poly = shrunk
                                else:
                                     # Fallback: try a smaller, fixed buffer distance if dynamic failed
                                     shrunk_fallback = bubble_polygon.buffer(-3.0, join_style=2)
                                     if shrunk_fallback.is_valid and not shrunk_fallback.is_empty and isinstance(shrunk_fallback, Polygon):
                                          print(f"   Debug: Used fallback buffer (-3.0) for bubble {i+1}")
                                          text_poly = shrunk_fallback
                           except Exception as buffer_err:
                                print(f"   ⚠️ Warning: Error shrinking polygon for bubble {i+1}: {buffer_err}. Using original polygon bounds for fitting.")
                                # If buffering fails critically, we might use the original polygon as a last resort,
                                # but find_optimal_text_settings should handle potential errors there.
                                text_poly = bubble_polygon # Less ideal, but try to continue

                           if not text_poly:
                                print(f"   ⚠️ Skipping bubble {i+1}: Could not create a valid polygon for text fitting.")
                                continue

                           # --- Find Optimal Text Settings ---
                           # Use the dummy draw object for measurement inside the function
                           text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                           if text_settings:
                                # --- Draw Text Layer ---
                                text_layer = draw_text_on_layer(text_settings, image_size)
                                if text_layer:
                                     # --- Composite Text Layer onto Main PIL Image ---
                                     try:
                                         # Paste the text layer (with transparency) onto the main image
                                         image_pil.paste(text_layer, (0, 0), text_layer)
                                         processed_count += 1
                                         print(f"   ✔️ Pasted text layer for bubble {i+1}")
                                     except Exception as paste_err:
                                         print(f"❌ Error pasting text layer for bubble {i+1}: {paste_err}")
                                else:
                                     print(f"   ⚠️ Text drawing function failed for bubble {i+1}, layer not created.")
                           else:
                                print(f"   ⚠️ Could not find optimal text settings for bubble {i+1}.")

                 except Exception as bubble_err:
                      # Catch unexpected errors during the processing of a single bubble
                      print(f"❌❌❌ Unhandled error processing bubble {i + 1}: {bubble_err}")
                      traceback.print_exc(limit=2)
                      emit_progress(3, f"Skipping bubble {i+1} due to error.", current_bubble_progress, sid) # Keep progress moving


             # --- Finalize image and result data AFTER the loop ---
             if mode == 'extract':
                 emit_progress(4, f"Finished extracting text ({processed_count}/{bubble_count}).", 95, sid)
                 final_image_np = inpainted_image # Final image is the cleaned one
                 output_filename = f"{output_filename_base}_cleaned.jpg"
                 result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
             elif mode == 'auto':
                 emit_progress(4, f"Finished drawing text ({processed_count}/{bubble_count}).", 95, sid)
                 if image_pil: # If PIL conversion and drawing occurred
                      try:
                           # Convert the final PIL image back to OpenCV BGR format for saving
                           final_image_np = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
                           output_filename = f"{output_filename_base}_translated.jpg"
                           result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
                      except Exception as convert_err:
                           print(f"❌ Error converting final PIL image back to CV2 format: {convert_err}. Saving cleaned image instead.")
                           emit_error("Failed to finalize translated image. Saving cleaned version.", sid)
                           final_image_np = inpainted_image # Fallback to saving cleaned image
                           output_filename = f"{output_filename_base}_cleaned_conversion_error.jpg"
                           result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []} # Treat as extract failure
                 else:
                      # If PIL conversion failed earlier, mode would be 'extract', but as a safeguard:
                      print("⚠️ Warning: 'auto' mode finished but PIL image wasn't available. Saving cleaned image.")
                      final_image_np = inpainted_image # Fallback to cleaned image
                      output_filename = f"{output_filename_base}_cleaned_pil_error.jpg"
                      result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}

             # Set the final output path (if not already set by an error condition)
             if not final_output_path:
                  final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)


        # === Step 5: Save Final Image ===
        # This step runs if processing completed (even partially) or if an error occurred
        # but we determined a fallback image (like the cleaned one) should be saved.
        if final_image_np is not None and final_output_path:
            emit_progress(5, "Saving final image...", 98, sid)
            save_success = False
            try:
                 save_success = cv2.imwrite(final_output_path, final_image_np)
                 if not save_success:
                     raise IOError(f"cv2.imwrite returned False for path: {final_output_path}")
                 print(f"✔️ Saved final image using OpenCV: {final_output_path}")
            except Exception as cv_save_err:
                 print(f"⚠️ OpenCV save failed: {cv_save_err}. Trying PIL fallback...")
                 try:
                     # Convert BGR (OpenCV) to RGB (PIL) before saving
                     pil_img_to_save = Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB))
                     # Ensure the directory exists one last time
                     os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                     pil_img_to_save.save(final_output_path)
                     save_success = True
                     print(f"✔️ Saved final image using PIL fallback: {final_output_path}")
                 except Exception as pil_save_err:
                      print(f"❌ PIL save also failed: {pil_save_err}")
                      emit_error("Failed to save final image using both OpenCV and PIL.", sid)
                      # Cannot proceed if saving fails

            # === Step 6: Signal Completion (only if save was successful) ===
            if save_success:
                processing_time = time.time() - start_time
                print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Final output: {final_output_path}")
                emit_progress(6, f"Processing complete ({processing_time:.2f}s).", 100, sid)
                # Ensure result_data has been populated
                if not result_data:
                    print("⚠️ Result data was empty before emitting completion. Creating default.")
                    result_data = {'mode': mode, 'imageUrl': f'/results/{os.path.basename(final_output_path)}'}
                    if mode == 'extract': result_data['translations'] = translations_list # Add translations if extract mode
                socketio.emit('processing_complete', result_data, room=sid) # Send results
            else:
                # If saving failed after trying both methods
                print(f"❌❌❌ Critical Error: Could not save final image for SID {sid} to {final_output_path}")
                # Error already emitted by the save block
        elif not final_output_path:
             # This case implies an error happened very early, before an output path was even determined.
             print(f"❌ SID {sid}: Processing aborted before final image path was set.")
             # Error should have been emitted previously.
        else: # final_image_np is None, but path might exist
            print(f"❌ SID {sid}: Processing ended with no final image data to save.")
            emit_error("Internal error: No final image generated.", sid)


    except Exception as e:
        # Catch-all for any unexpected errors within the main task function
        print(f"❌❌❌ UNHANDLED FATAL ERROR in process_image_task for SID {sid}: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        # Send a generic error message to the client
        emit_error(f"An unexpected server error occurred ({type(e).__name__}). Please check server logs or try again.", sid)

    finally:
        # --- Cleanup Uploaded File ---
        # This block executes whether the try block succeeded or failed
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                print(f"🧹 Cleaned up uploaded file: {image_path}")
        except Exception as cleanup_err:
            # Log cleanup errors but don't send to client as processing is already over/failed
            print(f"⚠️ Error cleaning up uploaded file {image_path}: {cleanup_err}")


# --- Flask Routes & SocketIO Handlers ---

@app.route('/')
def index():
    """ Serves the main HTML page. """
    return render_template('index.html')

@app.route('/results/<path:filename>') # Use path converter for flexibility
def get_result_image(filename):
    """ Serves processed images from the result folder. """
    # Basic security check: prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
    results_dir = os.path.abspath(app.config['RESULT_FOLDER'])
    try:
        return send_from_directory(results_dir, filename, as_attachment=False)
    except FileNotFoundError:
        return "File not found", 404

@socketio.on('connect')
def handle_connect():
    """ Handles new client connections. """
    print(f"✅ Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """ Handles client disconnections. """
    print(f"❌ Client disconnected: {request.sid}")

@socketio.on('start_processing')
def handle_start_processing(data):
    """
    Handles the 'start_processing' event from the client.
    Validates input, saves the uploaded file, and starts the background task.
    """
    sid = request.sid
    print(f"\n--- Received 'start_processing' event from SID: {sid} ---")

    # --- Robust input validation ---
    if not isinstance(data, dict):
        emit_error("Invalid request format (expected dictionary).", sid)
        return
    print(f"   Data keys received: {list(data.keys())}")

    file_data_str = data.get('file')
    mode = data.get('mode')

    if not file_data_str or not isinstance(file_data_str, str) or not file_data_str.startswith('data:image'):
        emit_error('Invalid or missing image data URI.', sid)
        return

    if mode not in ['extract', 'auto']:
        emit_error(f"Invalid or missing mode ('{mode}'). Must be 'extract' or 'auto'.", sid)
        return

    print(f"   Mode selected: '{mode}'")

    # --- File Saving ---
    upload_dir = app.config['UPLOAD_FOLDER']
    # Check if upload directory exists and is writable
    if not os.path.isdir(upload_dir) or not os.access(upload_dir, os.W_OK):
        print(f"❌ ERROR: Upload directory '{upload_dir}' is not accessible or writable.")
        emit_error("Server configuration error (cannot save uploads).", sid)
        return

    upload_path = None # Initialize path variable
    try:
        # Split the data URI (e.g., "data:image/jpeg;base64,/9j/...")
        header, encoded = file_data_str.split(',', 1)
        # Extract file extension (e.g., "jpeg")
        file_extension_match = re.search(r'data:image/(\w+)', header)
        if not file_extension_match:
             emit_error('Could not determine file type from data URI.', sid)
             return
        file_extension = file_extension_match.group(1).lower()

        # Validate extension against allowed list
        if file_extension not in ALLOWED_EXTENSIONS:
            emit_error(f'Invalid file type: {file_extension}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}.', sid)
            return

        # Decode base64 data
        file_bytes = base64.b64decode(encoded)
        if not file_bytes:
             emit_error('Empty file data after decoding.', sid); return

        # Check file size against limit (MAX_CONTENT_LENGTH is set in Flask config)
        if len(file_bytes) > app.config['MAX_CONTENT_LENGTH']:
             emit_error(f"File is too large (>{app.config['MAX_CONTENT_LENGTH'] // 1024 // 1024} MB).", sid); return

        # Generate unique filenames
        unique_id = uuid.uuid4()
        input_filename = f"{unique_id}.{file_extension}"
        output_filename_base = f"{unique_id}" # Base name for output files
        upload_path = os.path.join(upload_dir, input_filename)

        # Write the file
        with open(upload_path, 'wb') as f:
            f.write(file_bytes)
        print(f"   ✔️ File saved: {upload_path} ({len(file_bytes)/1024:.1f} KB)")

    except (ValueError, TypeError, IndexError) as decode_err:
        print(f"❌ Error decoding/parsing data URI: {decode_err}")
        emit_error('Error processing image data URI format.', sid)
        return
    except base64.binascii.Error as b64_err:
        print(f"❌ Base64 decoding error: {b64_err}")
        emit_error('Invalid Base64 encoding in image data.', sid)
        return
    except IOError as io_err:
         print(f"❌ File writing error: {io_err}")
         emit_error('Server error saving uploaded file.', sid)
         # Clean up partially saved file if it exists
         if upload_path and os.path.exists(upload_path):
             try:
                 os.remove(upload_path)
                 print(f"   🧹 Cleaned up partially saved file '{upload_path}' due to IO error.")
             except Exception as cleanup_err_io:
                 # Log error during cleanup but continue to return
                 print(f"   ⚠️ Error cleaning up file '{upload_path}' after IO error: {cleanup_err_io}")
         return # Important: return after handling the IO error
    except Exception as e:
        print(f"❌ Unexpected file handling error: {e}")
        traceback.print_exc()
        emit_error(f'Server error during file upload: {type(e).__name__}', sid)
        # Attempt cleanup even for unexpected errors if path was determined
        if upload_path and os.path.exists(upload_path):
             try:
                 os.remove(upload_path)
                 print(f"   🧹 Cleaned up file '{upload_path}' due to unexpected error during save.")
             except Exception as cleanup_err_unexp:
                 print(f"   ⚠️ Error cleaning up file '{upload_path}' after unexpected save error: {cleanup_err_unexp}")
        return

    # --- Start Background Task ---
    if upload_path: # Only proceed if file was saved successfully
        print(f"   Attempting to start background processing task...")
        try:
            # --- THIS IS THE CORRECTED CALL (using positional arguments) ---
            socketio.start_background_task(
                process_image_task,
                upload_path,            # 1st argument corresponds to 'image_path'
                output_filename_base,   # 2nd argument corresponds to 'output_filename_base'
                mode,                   # 3rd argument corresponds to 'mode'
                sid                     # 4th argument corresponds to 'sid'
            )
            # --- END OF CORRECTION ---

            print(f"   ✔️ Background task for SID {sid} initiated successfully.")
            # Notify client that processing has started
            socketio.emit('processing_started', {'message': 'Upload successful! Processing has started...'}, room=sid)

        except Exception as task_start_err:
            # Catch errors specifically related to *starting* the task
            print(f"❌ CRITICAL: Failed to start background task: {task_start_err}")
            traceback.print_exc()
            emit_error(f"Server error: Could not start image processing task ({task_start_err}).", sid)
            # Attempt to clean up the uploaded file if task failed to start
            if os.path.exists(upload_path):
                try:
                    os.remove(upload_path)
                    print(f"   🧹 Cleaned up file '{upload_path}' due to task start failure.")
                except Exception as cleanup_err_start:
                    print(f"   ⚠️ Error cleaning up file '{upload_path}' after task start failure: {cleanup_err_start}")

    else:
        # This case should theoretically not be reached if error handling above is correct
        print("❌ Internal Error: upload_path was not set before attempting to start task.")
        emit_error("Internal server error (upload path missing).", sid)

    print(f"--- Finished handling 'start_processing' request for SID: {sid} ---")


# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App ---")
    # Environment variable checks
    if not ROBOFLOW_API_KEY:
        print("███ WARNING: ROBOFLOW_API_KEY environment variable not set! Roboflow calls will fail. ███")
    if app.config['SECRET_KEY'] == 'change_this_in_production':
        print("⚠️ WARNING: Using default Flask SECRET_KEY. Set a strong secret in production environment variables!")

    # Determine port (Heroku/cloud providers often set PORT env var)
    port = int(os.environ.get('PORT', 5000))

    print(f"   * Flask App Name: {app.name}")
    print(f"   * Upload Folder: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"   * Result Folder: {os.path.abspath(app.config['RESULT_FOLDER'])}")
    print(f"   * Allowed Extensions: {ALLOWED_EXTENSIONS}")
    print(f"   * Text Formatter: {'Dummy' if isinstance(text_formatter, DummyTextFormatter) else 'Loaded'}")
    print(f"   * SocketIO Async Mode: {socketio.async_mode}")
    print(f"   * Roboflow Key Loaded: {'Yes' if ROBOFLOW_API_KEY else 'NO (!)'}")
    print(f"   * Starting server on http://0.0.0.0:{port}")

    # Run the SocketIO server
    # debug=False is recommended for production/eventlet
    # log_output=False silences default werkzeug logs if desired (can be noisy)
    try:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)
    except Exception as run_err:
         print(f"❌❌❌ Failed to start SocketIO server: {run_err}")
         sys.exit(1)
