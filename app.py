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
    # Basic check for essential functions (adjust if your names differ)
    if not all(hasattr(text_formatter, func) for func in ['set_arabic_font_path', 'get_font', 'format_arabic_text', 'layout_balanced_text']):
         print("⚠️ WARNING: 'text_formatter.py' seems to be missing required functions!")
         # Note: No ImportError raised here, allows running with partial module if needed
except ImportError as import_err:
    print(f"❌ ERROR: Cannot import 'text_formatter.py': {import_err}")
    class DummyTextFormatter: # Define dummy only if import fails
        def __init__(self): print("⚠️ WARNING: Initializing DummyTextFormatter.")
        def set_arabic_font_path(self, path): pass
        def get_font(self, size): return None
        def format_arabic_text(self, text): return text
        def layout_balanced_text(self, draw, text, font, target_width): return text
    text_formatter = DummyTextFormatter()
    print("⚠️ WARNING: Using dummy 'text_formatter'. Text functions will be basic or fail.")

load_dotenv()

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production_!!!')
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
    """Finds the font file path and sets it using the text_formatter object."""
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

# --- Constants ---
TEXT_COLOR = (0, 0, 0)
SHADOW_COLOR = (255, 255, 255)
SHADOW_OPACITY = 90
TRANSLATION_PROMPT = 'ترجم هذا النص داخل فقاعة المانجا إلى العربية بوضوح. أرجع الترجمة فقط بين علامتي اقتباس هكذا: "الترجمة هنا".'

# --- Helper Functions ---
def emit_progress(step, message, percentage, sid):
    socketio.emit('progress_update', {'step': step, 'message': message, 'percentage': percentage}, room=sid); socketio.sleep(0.01)
def emit_error(message, sid):
    print(f"SID: {sid} | ❗ ERROR Emitted: {message}"); socketio.emit('processing_error', {'error': message}, room=sid); socketio.sleep(0.01)

# --- Integrated Core Logic Functions ---

def get_roboflow_predictions(endpoint_url, api_key, image_b64, timeout=30):
    """Calls Roboflow API and returns predictions list."""
    model_name = endpoint_url.split('/')[-2] if '/' in endpoint_url else "Unknown"
    print(f"ℹ️ Calling Roboflow ({model_name})...")
    if not api_key: print("❌ ERROR: Roboflow API Key missing!"); raise ValueError("Missing Roboflow API Key.")
    try:
        response = requests.post(f"{endpoint_url}?api_key={api_key}", data=image_b64, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=timeout)
        response.raise_for_status()
        predictions = response.json().get("predictions", [])
        print(f"✔️ Roboflow ({model_name}) OK. Predictions: {len(predictions)}")
        return predictions
    except requests.exceptions.Timeout as err: print(f"❌ Roboflow ({model_name}) Timeout: {err}"); raise ConnectionError(f"Roboflow API ({model_name}) timed out.") from err
    except requests.exceptions.HTTPError as err: print(f"❌ Roboflow ({model_name}) HTTP Error: Status {err.response.status_code}"); print(f"   Response: {err.response.text[:200]}..."); raise ConnectionError(f"Roboflow API ({model_name}) failed (Status {err.response.status_code}). Check Key/URL.") from err
    except requests.exceptions.RequestException as err: print(f"❌ Roboflow ({model_name}) Request Error: {err}"); raise ConnectionError(f"Network error contacting Roboflow ({model_name}).") from err
    except Exception as err: print(f"❌ Roboflow ({model_name}) Unexpected Error: {err}"); traceback.print_exc(); raise RuntimeError(f"Unexpected error during Roboflow ({model_name}).") from err

def extract_translation(text):
    """Extracts text within the first pair of double quotes."""
    if not isinstance(text, str): return ""
    match = re.search(r'"(.*?)"', text, re.DOTALL); return match.group(1).strip() if match else text.strip().strip('"').strip()

def ask_luminai(prompt, image_bytes, max_retries=3, sid=None):
    """Sends request to LuminAI and extracts translation."""
    print("ℹ️ Calling LuminAI...")
    url = "https://luminai.my.id/"; payload = {"content": prompt, "imageBuffer": list(image_bytes), "options": {"clean_output": True}}; headers = {"Content-Type": "application/json", "Accept-Language": "ar"}
    for attempt in range(max_retries):
        print(f"   LuminAI Attempt {attempt + 1}/{max_retries}...")
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code == 200: translation = extract_translation(response.json().get("result", "")); print(f"✔️ LuminAI OK: '{translation[:50]}...'"); return translation
            elif response.status_code == 429: retry_after = int(response.headers.get('Retry-After', 5)); print(f"   ⚠️ LuminAI Rate limit (429). Retrying in {retry_after}s..."); socketio.sleep(retry_after)
            else: print(f"   ❌ LuminAI failed: Status {response.status_code} - {response.text[:100]}"); return ""
        except RequestException as err: print(f"   ❌ Network/Timeout error LuminAI (Attempt {attempt+1}): {err}");
        except Exception as err: print(f"   ❌ Unexpected error LuminAI (Attempt {attempt+1}): {err}"); traceback.print_exc(limit=1)
        if attempt == max_retries - 1: break; socketio.sleep(2 * (attempt + 1))
    print("   ❌ LuminAI failed after all retries."); return ""

def find_optimal_text_settings_final(draw, text, initial_shrunk_polygon):
    """
    *** PLACEHOLDER LOGIC *** - Based on original script structure.
    Replace/Refine with your specific layout requirements if needed.
    Searches for the best font size and text layout using text_formatter.
    """
    print("ℹ️ Finding optimal text settings (Placeholder/Original Logic)...") # Indicate potential placeholder
    if not initial_shrunk_polygon or initial_shrunk_polygon.is_empty or not initial_shrunk_polygon.is_valid or not text:
        print("   ⚠️ Invalid input to find_optimal_text_settings_final.")
        return None

    best_fit = None
    for font_size in range(65, 4, -1): # Iterate font sizes
        font = text_formatter.get_font(font_size)
        if font is None: continue # Skip size if font loading fails

        padding_distance = max(1.5, font_size * 0.12)
        try: # Shrink polygon for text fitting
            text_fitting_polygon = initial_shrunk_polygon.buffer(-padding_distance, join_style=2)
            if not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty: text_fitting_polygon = initial_shrunk_polygon.buffer(-2.0, join_style=2)
            if not isinstance(text_fitting_polygon, Polygon) or not text_fitting_polygon.is_valid or text_fitting_polygon.is_empty: continue
        except Exception as buffer_err: print(f"   ⚠️ Buffer error sz {font_size}: {buffer_err}"); continue

        minx, miny, maxx, maxy = text_fitting_polygon.bounds; target_width = maxx - minx; target_height = maxy - miny
        if target_width <= 5 or target_height <= 10: continue # Target area too small

        # --- Use text_formatter's layout function ---
        try: wrapped_text = text_formatter.layout_balanced_text(draw, text, font, target_width)
        except Exception as layout_err: print(f"   ⚠️ Layout error sz {font_size}: {layout_err}"); continue
        if not wrapped_text: continue

        # --- Measure and Check Fit ---
        try:
            m_bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, spacing=4, align='center')
            text_actual_width = m_bbox[2] - m_bbox[0]; text_actual_height = m_bbox[3] - m_bbox[1]; shadow_offset = max(1, font_size // 18)
            if (text_actual_height + shadow_offset) <= target_height and (text_actual_width + shadow_offset) <= target_width:
                x_offset = (target_width - text_actual_width) / 2; y_offset = (target_height - text_actual_height) / 2
                draw_x = minx + x_offset - m_bbox[0]; draw_y = miny + y_offset - m_bbox[1]
                best_fit = {'text': wrapped_text, 'font': font, 'x': int(draw_x), 'y': int(draw_y), 'font_size': font_size}
                print(f"   ✔️ Optimal fit found: Size={font_size}, Pos=({int(draw_x)},{int(draw_y)})"); break # Exit loop
        except Exception as measure_err: print(f"   ⚠️ Measure error sz {font_size}: {measure_err}"); continue

    if best_fit is None: print(f"   ⚠️ Warning: Could not find suitable font size for text: '{text[:30]}...'")
    return best_fit # Return found settings or None

def draw_text_on_layer(text_settings, image_size):
    """
    *** PLACEHOLDER LOGIC *** - Based on original script structure.
    Replace/Refine with your specific drawing requirements if needed.
    Draws text+shadow on a transparent layer using PIL.
    """
    print("ℹ️ Drawing text layer (Placeholder/Original Logic)...") # Indicate potential placeholder
    text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0)) # Start with empty layer
    if not text_settings or not isinstance(text_settings, dict): return text_layer
    try:
        draw_on_layer = ImageDraw.Draw(text_layer)
        font = text_settings.get('font'); text_to_draw = text_settings.get('text', ''); x = text_settings.get('x', 0); y = text_settings.get('y', 0); font_size = text_settings.get('font_size', 10)
        if not font or not text_to_draw: return text_layer
        shadow_offset = max(1, font_size // 18); shadow_color_with_alpha = SHADOW_COLOR + (SHADOW_OPACITY,)
        # Draw shadow
        draw_on_layer.multiline_text((x + shadow_offset, y + shadow_offset), text_to_draw, font=font, fill=shadow_color_with_alpha, align='center', spacing=4)
        # Draw text
        draw_on_layer.multiline_text((x, y), text_to_draw, font=font, fill=TEXT_COLOR + (255,), align='center', spacing=4)
        print(f"   ✔️ Drew text '{text_to_draw[:20]}...' at ({x},{y})")
    except Exception as e: print(f"❌ Error in draw_text_on_layer: {e}"); traceback.print_exc(limit=1)
    return text_layer


# --- Main Processing Task ---
def process_image_task(image_path, output_filename_base, mode, sid):
    """ Core logic: Cleans, optionally translates, and draws text based on mode. """
    start_time = time.time()
    inpainted_image = None; final_image_np = None; translations_list = []
    final_output_path = ""; result_data = {}
    try:
        # === Step 0: Load Image ===
        emit_progress(0, "Loading image...", 5, sid)
        image = cv2.imread(image_path)
        if image is None: raise ValueError(f"Load failed: {image_path}")
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] != 3: raise ValueError("Unsupported image format")
        h_img, w_img = image.shape[:2]; original_image_for_cropping = image.copy(); result_image = image.copy()

        # === Step 1: Remove Text (Inpainting) ===
        emit_progress(1, "Detecting text regions...", 10, sid)
        _, buffer = cv2.imencode('.jpg', image); b64_image = base64.b64encode(buffer).decode('utf-8') if buffer is not None else None
        if not b64_image: raise ValueError("Encoding failed (Text Detect)")
        text_predictions = []
        try: text_predictions = get_roboflow_predictions('https://serverless.roboflow.com/text-detection-w0hkg/1', ROBOFLOW_API_KEY, b64_image)
        except Exception as rf_err: print(f"⚠️ Text detect failed: {rf_err}"); emit_progress(1, f"Text detection failed, skipping.", 15, sid)
        emit_progress(1, f"Masking {len(text_predictions)} areas...", 15, sid)
        text_mask = np.zeros(image.shape[:2], dtype=np.uint8); polygons_drawn = 0
        for pred in text_predictions: # Mask drawing loop
             points = pred.get("points", [])
             if len(points) >= 3:
                 polygon_np = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
                 polygon_np[:, 0]=np.clip(polygon_np[:, 0],0,w_img-1); polygon_np[:, 1]=np.clip(polygon_np[:, 1],0,h_img-1)
                 try: cv2.fillPoly(text_mask, [polygon_np], 255); polygons_drawn += 1
                 except Exception as fill_err: print(f"⚠️ Warn: Draw text poly err: {fill_err}")

        if np.any(text_mask) and polygons_drawn > 0:
            emit_progress(1, f"Inpainting {polygons_drawn} areas...", 20, sid)
            # --- CORRECTED try...except block for inpainting ---
            try:
                print("   Attempting cv2.inpaint...") # Add log
                inpainted_image = cv2.inpaint(result_image, text_mask, 10, cv2.INPAINT_NS) # Use INPAINT_NS
                if inpainted_image is None:
                    print("   ❌ ERROR: cv2.inpaint returned None!")
                    raise RuntimeError("cv2.inpaint failed and returned None")
                print("   ✔️ cv2.inpaint successful.")
                emit_progress(1, "Inpainting complete.", 25, sid)
            except Exception as inpaint_err:
                # This except block must be at the SAME indentation level as the 'try' above
                print(f"❌ Error during inpainting: {inpaint_err}")
                traceback.print_exc(limit=1)
                emit_error(f"Inpainting failed: {inpaint_err}", sid)
                inpainted_image = result_image.copy() # Fallback
                print("   Using original image as fallback due to inpainting error.")
            # --- END of corrected try...except block ---
        else:
            emit_progress(1, "No text masked/found.", 25, sid); inpainted_image = result_image.copy()


        # === Step 2: Detect Bubbles ===
        emit_progress(2, "Detecting speech bubbles...", 30, sid)
        _, buffer_bubble = cv2.imencode('.jpg', inpainted_image); b64_bubble = base64.b64encode(buffer_bubble).decode('utf-8') if buffer_bubble is not None else None
        if not b64_bubble: raise ValueError("Encoding failed (Bubble Detect)")
        bubble_predictions = []
        try: bubble_predictions = get_roboflow_predictions('https://outline.roboflow.com/yolo-0kqkh/2', ROBOFLOW_API_KEY, b64_bubble)
        except Exception as rf_err:
             print(f"❌ Bubble detection failed: {rf_err}."); emit_error(f"Bubble detection failed ({type(rf_err).__name__}).", sid)
             final_image_np = inpainted_image; output_filename = f"{output_filename_base}_cleaned_nobubbles.jpg"; final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
             result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
             raise StopIteration("Bubble detection failed, saving cleaned image") # Use custom signal

        emit_progress(2, f"Bubble Detection (Found {len(bubble_predictions)} bubbles)...", 40, sid)

        # === Step 3 & 4: Process Bubbles & Finalize ===
        if not bubble_predictions:
             emit_progress(4, "No bubbles detected.", 95, sid); final_image_np = inpainted_image; output_filename = f"{output_filename_base}_cleaned.jpg"
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename); result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': []}
        else:
             image_pil = None; image_size = (w_img, h_img); temp_draw_for_settings = None
             if mode == 'auto':
                 try: image_pil = Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)).convert('RGBA'); image_size = image_pil.size; temp_draw_for_settings = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
                 except Exception as pil_conv_err: emit_error(f"Cannot draw (ImgConvFail: {pil_conv_err})", sid); print(f"⚠️ PIL fail, fallback to extract."); mode = 'extract'

             bubble_count = len(bubble_predictions); processed_count = 0; base_progress = 45; max_progress_bubbles = 90
             for i, pred in enumerate(bubble_predictions): # Bubble processing loop
                 current_bubble_progress = base_progress + int((i / bubble_count) * (max_progress_bubbles - base_progress))
                 emit_progress(3, f"Processing bubble {i + 1}/{bubble_count}...", current_bubble_progress, sid)
                 try:
                      points = pred.get("points", []); coords=[(int(p["x"]),int(p["y"])) for p in points]
                      if len(points) < 3: print(f"   Skip bbl {i+1}: <3 points"); continue
                      bubble_polygon = Polygon(coords) # Validate polygon
                      if not bubble_polygon.is_valid: bubble_polygon = make_valid(bubble_polygon)
                      if bubble_polygon.geom_type=='MultiPolygon': bubble_polygon = max(bubble_polygon.geoms, key=lambda p:p.area,default=None)
                      if not isinstance(bubble_polygon, Polygon) or bubble_polygon.is_empty: print(f"   Skip bbl {i+1}: Invalid polygon"); continue

                      minx, miny, maxx, maxy = map(int, bubble_polygon.bounds) # Crop original
                      minx_c=max(0,minx-5); miny_c=max(0,miny-5); maxx_c=min(w_img,maxx+5); maxy_c=min(h_img,maxy+5)
                      if maxx_c<=minx_c or maxy_c<=miny_c: print(f"   Skip bbl {i+1}: Invalid crop dims"); continue
                      bubble_crop = original_image_for_cropping[miny_c:maxy_c, minx_c:maxx_c]
                      if bubble_crop.size == 0: print(f"   Skip bbl {i+1}: Empty crop area"); continue

                      # --- CORRECTED Encoding Check ---
                      print(f"   Encoding bubble crop {i+1}...")
                      retval, crop_buffer_enc = cv2.imencode('.jpg', bubble_crop) # Get retval
                      if not retval: print(f"   ⚠️ Failed encode crop {i+1}. Skip."); continue
                      crop_bytes = crop_buffer_enc.tobytes() # Assign ONLY if encode succeeded
                      # --- End Corrected Check ---

                      translation = ask_luminai(TRANSLATION_PROMPT, crop_bytes, sid=sid)
                      if not translation: translation = "[ترجمة فارغة]"

                      if mode == 'extract':
                           translations_list.append({'id': i + 1, 'translation': translation}); processed_count += 1
                      elif mode == 'auto' and image_pil:
                           if translation == "[ترجمة فارغة]": continue
                           if isinstance(text_formatter, DummyTextFormatter): print("⚠️ Skip draw (DummyFormatter)"); continue

                           arabic_text = text_formatter.format_arabic_text(translation)
                           if not arabic_text: print(f"   Skip bbl {i+1}: Empty formatted text"); continue

                           poly_width=maxx-minx; poly_height=maxy-miny # Shrink polygon
                           initial_buffer_distance = max(3.0, (poly_width + poly_height) / 2 * 0.10)
                           text_poly = bubble_polygon # Default
                           try: shrunk = bubble_polygon.buffer(-initial_buffer_distance, join_style=2)
                                if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type=='Polygon': text_poly = shrunk
                                else: shrunk = bubble_polygon.buffer(-3.0, join_style=2);
                                      if shrunk.is_valid and not shrunk.is_empty and shrunk.geom_type=='Polygon': text_poly = shrunk
                           except Exception as buffer_err: print(f"⚠️ Warn: buffer error {buffer_err}")

                           # --- Call layout function (potentially placeholder) ---
                           text_settings = find_optimal_text_settings_final(temp_draw_for_settings, arabic_text, text_poly)

                           if text_settings:
                                # --- Call drawing function (potentially placeholder) ---
                                text_layer = draw_text_on_layer(text_settings, image_size)
                                if text_layer:
                                     try: image_pil.paste(text_layer, (0, 0), text_layer); processed_count += 1
                                     except Exception as paste_err: print(f"❌ Paste Error bbl {i+1}: {paste_err}")
                                else: print(f"⚠️ Draw func failed bbl {i+1}")
                           else: print(f"⚠️ Text fit failed bbl {i+1}")

                 except Exception as bubble_err: print(f"❌ Error bubble {i + 1}: {bubble_err}"); traceback.print_exc(limit=1); emit_progress(3, f"Skip bbl {i+1} (err).", current_bubble_progress + 1, sid)
             # End bubble loop

             # Finalize based on mode after loop
             if mode == 'extract':
                 emit_progress(4, f"Finished extract ({processed_count}/{bubble_count}).", 95, sid); final_image_np = inpainted_image; output_filename = f"{output_filename_base}_cleaned.jpg"; result_data = {'mode': 'extract', 'imageUrl': f'/results/{output_filename}', 'translations': translations_list}
             elif mode == 'auto':
                 emit_progress(4, f"Finished drawing ({processed_count}/{bubble_count}).", 95, sid); final_image_np = inpainted_image # Default
                 if image_pil: try: final_image_np = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR); except Exception as conv_e: print(f"❌ PIL->CV2 err: {conv_e}")
                 output_filename = f"{output_filename_base}_translated.jpg"; result_data = {'mode': 'auto', 'imageUrl': f'/results/{output_filename}'}
             final_output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)


        # === Step 5: Save Final Image ===
        emit_progress(5, "Saving final image...", 98, sid)
        if final_image_np is None: raise RuntimeError("Final image None before save.")
        if not final_output_path: raise RuntimeError("Final output path not set.")
        save_success = False
        try: save_success = cv2.imwrite(final_output_path, final_image_np);
             if not save_success: raise IOError("cv2.imwrite returned false")
        except Exception as cv_save_err:
             print(f"⚠️ OpenCV save failed: {cv_save_err}. Try PIL...");
             try: Image.fromarray(cv2.cvtColor(final_image_np, cv2.COLOR_BGR2RGB)).save(final_output_path); save_success = True; print("✔️ Saved via PIL.")
             except Exception as pil_save_err: print(f"❌ PIL save also failed: {pil_save_err}"); raise IOError(f"Save fail (CV/PIL)") from pil_save_err

        # === Step 6: Signal Completion ===
        processing_time = time.time() - start_time
        print(f"✔️ SID {sid} Processing complete in {processing_time:.2f}s. Output: {final_output_path}")
        emit_progress(6, f"Complete ({processing_time:.2f}s).", 100, sid)
        socketio.emit('processing_complete', result_data, room=sid) # Send results

    except StopIteration as stop_signal: # Catch early exit signal
        print(f"ℹ️ Task exited early SID {sid}: {stop_signal}")
        # Saving and completion signal should have happened before raising StopIteration
    except Exception as e:
        print(f"❌❌❌ Unhandled error task SID {sid}: {e}"); traceback.print_exc(); emit_error(f"Unexpected server error ({type(e).__name__}).", sid)
    finally: # Cleanup Uploaded File
        try:
            if image_path and os.path.exists(image_path): os.remove(image_path); print(f"🧹 Cleaned up: {image_path}")
        except Exception as cleanup_err: print(f"⚠️ Error cleaning up {image_path}: {cleanup_err}")


# --- Flask Routes & SocketIO Handlers ---
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
    sid = request.sid; print(f"\n--- Received 'start_processing' from SID: {sid} ---")
    if not isinstance(data, dict): emit_error("Invalid request format.", sid); return
    print(f"   Data keys: {list(data.keys())}")
    if 'file' not in data or not isinstance(data['file'], str) or not data['file'].startswith('data:image'): emit_error('Invalid/missing file data.', sid); return
    if 'mode' not in data or data['mode'] not in ['extract', 'auto']: emit_error('Invalid/missing mode.', sid); return
    mode = data['mode']; print(f"   Mode: '{mode}'")
    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir) or not os.access(upload_dir, os.W_OK): emit_error("Server upload dir error.", sid); return
    upload_path = None
    try:
        file_data_str = data['file']; header, encoded = file_data_str.split(',', 1)
        file_extension = header.split('/')[1].split(';')[0].split('+')[0]
        if file_extension not in ALLOWED_EXTENSIONS: emit_error(f'Invalid file type: {file_extension}.', sid); return
        file_bytes = base64.b64decode(encoded); print(f"   Decoded Size: {len(file_bytes)/1024:.1f} KB")
        unique_id = uuid.uuid4(); input_filename = f"{unique_id}.{file_extension}"; output_filename_base = f"{unique_id}"
        upload_path = os.path.join(upload_dir, input_filename)
        with open(upload_path, 'wb') as f: f.write(file_bytes); print(f"   ✔️ File saved: {upload_path}")
    except Exception as e: print(f"❌ File handling error: {e}"); traceback.print_exc(); emit_error(f'Server upload error: {type(e).__name__}', sid); return
    if upload_path:
        print(f"   Attempting task start...")
        try: socketio.start_background_task(process_image_task, upload_path, output_filename_base, mode, sid); print(f"   ✔️ Task initiated."); socketio.emit('processing_started', {'message': 'Upload OK! Processing...'}, room=sid)
        except Exception as task_err: print(f"❌ Failed start task: {task_err}"); traceback.print_exc(); emit_error(f"Server error starting task: {task_err}", sid);
            if os.path.exists(upload_path): try: os.remove(upload_path); print(f"   🧹 Cleaned up (task start fail).") except Exception as cl_e: print(f"   ⚠️ Cleanup failed: {cl_e}")
    else: emit_error("Server error (upload path missing).", sid)
    print(f"--- Finished handling 'start_processing' for SID: {sid} ---")

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Manga Processor Web App ---")
    if not ROBOFLOW_API_KEY: print("⚠️ WARNING: ROBOFLOW_API_KEY not set!")
    if app.config['SECRET_KEY'] == 'change_this_in_production_!!!': print("⚠️ WARNING: Using default FLASK_SECRET_KEY!")
    port = int(os.environ.get('PORT', 5000))
    print(f"   * Ready on http://0.0.0.0:{port}"); socketio.run(app, host='0.0.0.0', port=port, debug=False, log_output=False)

