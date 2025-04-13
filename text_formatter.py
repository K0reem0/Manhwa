# text_formatter.py
# -*- coding: utf-8 -*-

from PIL import ImageDraw, ImageFont
import arabic_reshaper
import os

# --- وظائف مساعدة للخط والنص العربي (يمكن نسخها أو استيرادها) ---

font_cache = {}
arabic_font_path = None # سيتم تحديده عند الاستدعاء أو البحث عنه هنا

def set_arabic_font_path(path):
    """Sets the path for the Arabic font."""
    global arabic_font_path
    if path and os.path.exists(path):
        arabic_font_path = path
        print(f"ℹ️ text_formatter using Arabic font: {arabic_font_path}")
    else:
        print(f"⚠️ text_formatter: Arabic font path '{path}' not valid. Using default.")
        arabic_font_path = None # Reset if invalid path given

def get_font(size):
    """يحصل على كائن خط بالحجم المحدد، مع استخدام ذاكرة التخزين المؤقت."""
    global font_cache, arabic_font_path
    size = max(1, int(size))
    if size not in font_cache:
        try:
            if arabic_font_path: # No need to check exists again if set via set_arabic_font_path
                 font_cache[size] = ImageFont.truetype(arabic_font_path, size)
            else:
                 # Use default PIL font if no custom path is set or valid
                 font_cache[size] = ImageFont.load_default()
        except Exception as e:
             print(f"⚠️ Error loading font size {size}: {e}. Falling back to default.")
             try:
                 font_cache[size] = ImageFont.load_default()
             except Exception as pil_e:
                 print(f"❌ CRITICAL: Failed to load default PIL font: {pil_e}")
                 return None # Cannot proceed without any font

    # Ensure the loaded object is valid
    if size not in font_cache or not hasattr(font_cache[size], 'getbbox'):
         print(f"⚠️ Warning: Font object for size {size} seems invalid. Using fallback.")
         try:
             font_cache[size] = ImageFont.load_default()
             if not hasattr(font_cache[size], 'getbbox'): return None
         except Exception:
             return None
    return font_cache.get(size)


def measure_text_width(draw, text, font):
    """يقيس عرض النص باستخدام Pillow."""
    if not text or not font or not draw: return 0
    try:
        # الطريقة المفضلة
        return draw.textlength(text, font=font)
    except AttributeError:
        # طريقة بديلة
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0]
        except Exception:
             # تقدير تقريبي
             font_size = getattr(font, 'size', 10)
             return len(text) * font_size * 0.6 # Adjust factor as needed
    except Exception:
        # تقدير تقريبي
        font_size = getattr(font, 'size', 10)
        return len(text) * font_size * 0.6

def format_arabic_text(text):
    """يعيد تشكيل النص العربي لضمان العرض الصحيح."""
    if isinstance(text, str):
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            # Optional: Bidi handling if needed for complex cases
            # from bidi.algorithm import get_display
            # return get_display(reshaped_text)
            return reshaped_text
        except Exception as e:
            print(f"⚠️ Error reshaping Arabic text: {e}. Returning original.")
            return text
    return ""

# --- وظيفة التنسيق الجديدة والمحسنة ---

def layout_balanced_text(draw, text, font, target_width):
    """
    محاولة لتنسيق النص بشكل متوازن (هدف مشابه لـ JSX shaper).
    يستخدم قياس العرض من Pillow.
    """
    words = text.split()
    if not words or not font or target_width <= 0:
        return ""

    lines = []
    current_line_words = []
    word_index = 0

    # --- المرحلة 1: التوزيع الأولي (Greedy Wrapping) ---
    # توزيع الكلمات على الأسطر بناءً على العرض المستهدف
    initial_lines = []
    temp_line_words = []
    idx = 0
    while idx < len(words):
        word = words[idx]
        potential_line_words = temp_line_words + [word]
        potential_line = " ".join(potential_line_words)
        line_width = measure_text_width(draw, potential_line, font)

        if not temp_line_words and line_width > target_width:
            # كلمة واحدة أطول من السطر
            initial_lines.append(word)
            temp_line_words = []
            idx += 1
        elif line_width <= target_width:
            # الكلمة تناسب السطر الحالي
            temp_line_words = potential_line_words
            idx += 1
            if idx == len(words): # آخر كلمة في النص
                 initial_lines.append(" ".join(temp_line_words))
        else:
            # الكلمة التالية لا تناسب، أكمل السطر الحالي
            if temp_line_words:
                initial_lines.append(" ".join(temp_line_words))
            temp_line_words = [word] # ابدأ سطرًا جديدًا بالكلمة الحالية
            idx += 1
            if idx == len(words): # إذا كانت هذه الكلمة هي الأخيرة
                 initial_lines.append(" ".join(temp_line_words))
                 temp_line_words = [] # أمان إضافي

    # --- المرحلة 2: محاولة الموازنة ---
    # نقل الكلمات بين الأسطر المتجاورة لتحسين التوازن البصري.
    # هذه خوارزمية تقريبية ومبسطة مقارنة بـ JSX.

    final_lines = list(initial_lines)
    if len(final_lines) <= 1:
        return "\n".join(final_lines) # لا حاجة للموازنة

    # قم بتكرار عملية الموازنة عدة مرات (لتسمح للتغييرات بالانتشار)
    for _pass in range(min(5, len(final_lines))): # عدد مرات محدود
        made_change = False
        # المرور على أزواج الأسطر المتجاورة
        for i in range(len(final_lines) - 1):
            line1_str = final_lines[i]
            line2_str = final_lines[i+1]
            words1 = line1_str.split()
            words2 = line2_str.split()

            if not words1 or not words2: continue # تخطي الأسطر الفارغة

            width1 = measure_text_width(draw, line1_str, font)
            width2 = measure_text_width(draw, line2_str, font)

            # الحالة 1: محاولة نقل آخر كلمة من السطر الأول إلى بداية السطر الثاني
            if len(words1) > 1: # لا تنقل إذا كانت الكلمة الوحيدة
                last_word_l1 = words1[-1]
                candidate_l1 = " ".join(words1[:-1])
                candidate_l2 = last_word_l1 + " " + line2_str
                cand_width1 = measure_text_width(draw, candidate_l1, font)
                cand_width2 = measure_text_width(draw, candidate_l2, font)

                # معيار التحسين: هل السطر الثاني الجديد مناسب؟ وهل الفروقات في العرض تتحسن؟
                # (نحاول جعل الأسطر أقرب لبعضها في الطول)
                original_diff = abs(width1 - width2)
                candidate_diff = abs(cand_width1 - cand_width2)

                # الشرط: السطر الثاني الجديد لا يتجاوز العرض، والفارق الجديد أصغر بشكل ملحوظ
                if cand_width2 <= target_width and candidate_diff < original_diff * 0.85:
                    final_lines[i] = candidate_l1
                    final_lines[i+1] = candidate_l2
                    made_change = True
                    # print(f"Pass {_pass+1}, L{i+1}->L{i+2}: Moved '{last_word_l1}' back") # For debugging
                    continue # انتقل للزوج التالي بعد التغيير

            # الحالة 2: محاولة نقل أول كلمة من السطر الثاني إلى نهاية السطر الأول
            # (أعد حساب العرض الأصلي لأن الحالة 1 قد تكون غيرت الأسطر)
            line1_str = final_lines[i]
            line2_str = final_lines[i+1]
            words1 = line1_str.split()
            words2 = line2_str.split()
            if not words1 or not words2: continue
            width1 = measure_text_width(draw, line1_str, font)
            width2 = measure_text_width(draw, line2_str, font)

            if len(words2) > 0: # تأكد من وجود كلمة لنقلها
                first_word_l2 = words2[0]
                candidate_l1 = line1_str + " " + first_word_l2
                candidate_l2 = " ".join(words2[1:])
                cand_width1 = measure_text_width(draw, candidate_l1, font)
                cand_width2 = measure_text_width(draw, candidate_l2, font) if candidate_l2 else 0 # عرض صفر لسطر فارغ

                original_diff = abs(width1 - width2)
                candidate_diff = abs(cand_width1 - cand_width2)

                # الشرط: السطر الأول الجديد لا يتجاوز العرض، والفارق الجديد أصغر بشكل ملحوظ
                # تأكد من أن السطر الثاني لن يصبح فارغًا إلا إذا كان السطر الأول قصيرًا جدًا
                can_move = True
                if not candidate_l2 and width1 > target_width * 0.6: # لا تجعل السطر الثاني فارغًا إذا كان الأول طويلاً نسبيًا
                    can_move = False

                if can_move and cand_width1 <= target_width and candidate_diff < original_diff * 0.85:
                    final_lines[i] = candidate_l1
                    final_lines[i+1] = candidate_l2
                    # إزالة الأسطر الفارغة المحتملة
                    final_lines = [line for line in final_lines if line]
                    made_change = True
                    # print(f"Pass {_pass+1}, L{i+2}->L{i+1}: Moved '{first_word_l2}' forward") # For debugging
                    continue # انتقل للزوج التالي

        if not made_change:
            # print(f"Balancing finished after pass {_pass+1}") # For debugging
            break # توقف عن التكرار إذا لم تحدث تغييرات

    # إزالة أي أسطر فارغة قد تنتج عن العملية
    final_lines = [line for line in final_lines if line.strip()]
    return "\n".join(final_lines)
