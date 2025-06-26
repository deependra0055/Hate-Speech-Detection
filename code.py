# import libraries
import streamlit as st
import whisper
import tempfile
import os
import openai
import re

# Seting up OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Title
st.title("🎙️ Hate Speech Detection from Audio using GenAI")
st.markdown("Upload an audio clip. The app will transcribe the audio and detect if it contains hate speech using GPT-4o.")

# Upload audio file
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        temp_audio_path = tmp.name

    st.audio(audio_file, format="audio/wav")

    # Transcribe using Whisper
    st.info("🔄 Transcribing audio...")
    result = model.transcribe(temp_audio_path)
    transcription = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcription)

    # Text preprocessing
    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
        return text.lower().strip()

    cleaned_text = clean_text(transcription)

    # OpenAI GPT-4o for hate speech detection
    st.info("🤖 Detecting hate speech using GPT-4o...")

    prompt = f"""
You are a hate speech detection AI. Given the following text, classify it as:
1. Hate Speech
2. Not Hate Speech

Also provide a short reasoning for your decision.

Text:
\"{cleaned_text}\"
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a hate speech detection expert."},
                {"role": "user", "content": prompt}
            ]
        )

        result_text = response["choices"][0]["message"]["content"]

        st.subheader("📊 Detection Result")
        st.success(result_text)

    except Exception as e:
        st.error(f"Error from OpenAI: {e}")

    # Clean up temp file
    os.remove(temp_audio_path)
