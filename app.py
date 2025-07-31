
import gradio as gr
import requests
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import os

# 🔐 Use environment variable for security
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "mistralai/mistral-7b-instruct"

def extract_text_from_file(file):
    if file is None:
        return ""

    ext = os.path.splitext(file.name)[-1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    elif ext == ".pdf":
        text = ""
        with fitz.open(file.name) as doc:
            for page in doc:
                text += page.get_text()
        return text
    else:
        return "❌ Unsupported file type. Please upload PDF or image."

def ask_medical_assistant(message, history, language):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    language_hint = {
        "English": "",
        "Hindi": "Please respond in Hindi."
    }

    messages = [{"role": "system", "content": (
        "You are a helpful medical assistant who explains lab tests and results clearly. "
        "Do not give diagnoses. Remind users to consult a doctor.")}]
    messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"{message}\n\n{language_hint.get(language, '')}"
    })

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        reply = f"⚠️ API Error: {str(e)}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    return convert_to_styled_messages(history), history, ""

def handle_file(file, history, language):
    extracted = extract_text_from_file(file)
    
    # Ensure it's a string
    if not isinstance(extracted, str) or not extracted.strip():
        return [{"role": "assistant", "content": "⚠️ Could not extract any text."}], history, ""

    return ask_medical_assistant(extracted, history, language)

def clear_chat():
    return [], [], ""

def convert_to_styled_messages(history):
    messages = []
    for i in range(0, len(history), 2):
        user_msg = history[i]["content"] if i < len(history) else ""
        assistant_msg = history[i + 1]["content"] if i + 1 < len(history) else ""

        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})

    return messages

# 🌟 UI with styling
with gr.Blocks(css="""
    .gradio-container {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
        padding: 10px;
    }
    .title {
        text-align: center;
        font-size: 28px;
        margin-bottom: 10px;
    }
    .chatbot-container {
        max-height: 500px;
    }
""") as demo:
    gr.Markdown("<div class='title'>🩺 Medical Test Assistant (Multilingual)</div>")

    with gr.Row():
        language = gr.Dropdown(["English", "Hindi"], value="English", label="🌐 Select Language", scale=1)
        clear_btn = gr.Button("🧹 Clear Chat", scale=0)

    with gr.Row():
        chatbot = gr.Chatbot(
            label="",
            height=460,
            show_copy_button=True,
            scale=3,
            type="messages"
        )
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="📎 Upload Report (PDF/Image)",
                file_types=[".pdf", ".png", ".jpg", ".jpeg"],
            )

    msg = gr.Textbox(
        placeholder="💬 Type your question here and press Enter...",
        show_label=False,
        lines=2
    )

    state = gr.State([])

    msg.submit(ask_medical_assistant, [msg, state, language], [chatbot, state, msg])
    file_upload.change(handle_file, [file_upload, state, language], [chatbot, state, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, state, msg])


demo.launch(server_name="0.0.0.0", server_port=10000)
