import streamlit as st
import whisper
import tempfile
import os

# Initialize session state variables
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Function to load the Whisper model
def load_model(model_name):
    st.session_state.model = whisper.load_model(model_name)
    st.session_state.model_loaded = True
    st.sidebar.success(f"Whisper Model '{model_name}' Loaded")

# Title of the app
st.title("Whisper Transcription App")

# File uploader for audio
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

# Model selection
model_options = ["tiny", "base", "small", "medium", "large"]
model_name = st.sidebar.selectbox("Select Whisper Model", model_options)

# Load model button
if st.sidebar.button("Load Model"):
    load_model(model_name)

# Transcribe button (shown only after model is loaded)
if st.session_state.model_loaded:
    if st.sidebar.button("Transcribe Audio"):
        if audio_file is not None:
            st.sidebar.success("Transcribing Audio")

            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio_file.read())
                temp_audio_path = temp_audio_file.name

            try:
                # Transcribe the audio
                transcription = st.session_state.model.transcribe(temp_audio_path)

                st.sidebar.success("Transcription Complete")
                transcribed_text = transcription["text"]
                st.markdown(transcribed_text)

                # Provide download option for the transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcribed_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
            finally:
                # Remove the temporary file
                os.remove(temp_audio_path)
        else:
            st.sidebar.error("Please upload an audio file")

# Option to play the original audio file
st.sidebar.header("Play Original Audio File")
if audio_file:
    st.sidebar.audio(audio_file)
else:
    st.sidebar.info("Please load a model to proceed with transcription.")
