import gradio as gr
import pytesseract
import cv2
from PIL import Image
import numpy as np
from deep_translator import GoogleTranslator
import os

# 🛠️ Install tesseract-ocr in runtime (first-time only)
os.system("apt-get update && apt-get install -y tesseract-ocr")

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(gray, lang='eng')
        return text
    except Exception as e:
        return f"Image processing error: {str(e)}"

def extract_text_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    extracted_text = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(frame_rgb, lang='eng')
        extracted_text += text + "\n"

    cap.release()
    return extracted_text

def translate_text(text, target_lang="hi"):
    if not text.strip():
        return "No text found.", ""
    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        return text, translated
    except Exception as e:
        return text, f"Translation error: {str(e)}"

def process_input(file, lang):
    try:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            extracted = extract_text_from_image(file)
        elif ext in ['.mp4', '.avi', '.mov']:
            extracted = extract_text_from_video(file.name)
        else:
            return "Unsupported file format.", ""
        return translate_text(extracted, lang)
    except Exception as e:
        return "Error processing file", f"Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("## 📸 English Text Extractor & Translator (Image/Video)")
    with gr.Row():
        file_input = gr.File(label="Upload Image or Video")
        lang_input = gr.Textbox(label="Target Language Code (e.g., hi, te, fr)")
    with gr.Row():
        eng_text = gr.Textbox(label="Extracted English Text")
        trans_text = gr.Textbox(label="Translated Text")
    btn = gr.Button("Translate")
    btn.click(fn=process_input, inputs=[file_input, lang_input], outputs=[eng_text, trans_text])

demo.launch()
