import os
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import io
import speech_recognition as sr
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pyttsx3

UPLOAD_FOLDER = 'uploaded_audio'
RESPONSE_FOLDER = 'responses'
TRANSCRIPT_FILE = os.path.join(RESPONSE_FOLDER, 'transcript.txt')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESPONSE_FOLDER, exist_ok=True)

# Load the trained model
model = load_model('emotion_recognition_model.h5')

emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
encoder = OneHotEncoder()
encoder.fit(np.array(emotion_labels).reshape(-1, 1))

def preprocess_audio(audio_path):
    data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    features = np.hstack((
        np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0),
        np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(data)), sr=sr).T, axis=0),
        np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0),
        np.mean(librosa.feature.rms(y=data).T, axis=0),
        np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    ))
    return np.expand_dims(features, axis=(0, 2))

def record_audio():
    fs, seconds = 44100, 5
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    path = os.path.join(UPLOAD_FOLDER, 'recording.wav')
    wav.write(path, fs, recording)
    return path

def convert_speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        try:
            return recognizer.recognize_google(recognizer.record(source))
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the speech recognition service."

def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        audio_file = os.path.join(RESPONSE_FOLDER, 'response.wav')
        engine.save_to_file(text, audio_file)
        engine.runAndWait()
        with open(audio_file, 'rb') as f:
            return f.read()
    except Exception as e:
        st.error(f"Text-to-Speech error: {e}")
        return b''

def play_audio(audio_data):
    st.audio(io.BytesIO(audio_data), format='audio/wav')

def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)
    scaler = StandardScaler()
    features = scaler.fit_transform(features.reshape(-1, features.shape[1])).reshape(features.shape)
    return features

def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def predict_emotion(file_path):
    features = get_features(file_path)
    predictions = model.predict(features)
    avg_predictions = np.mean(predictions, axis=0)
    predicted_label = np.argmax(avg_predictions)
    return emotion_labels[predicted_label]

def handle_emotion_based_response(emotion):
    responses = {
        'happy': "I'm glad to hear that!",
        'sad': "I'm sorry to hear that. Is there anything I can do to help?",
        'angry': "I understand you're upset. Let's try to resolve this.",
        'fear': "I sense fear. Let's try to address your concerns.",
        'disgust': "I understand your concern. Let's see how we can resolve it.",
        'surprise': "That's surprising! Tell me more.",
        'neutral': "How can I assist you today?"
    }
    return responses.get(emotion, "How can I assist you today?")

def handle_audio(file_path):
    transcript = convert_speech_to_text(file_path)
    emotion = predict_emotion(file_path)
    if is_critical_emotion(emotion):
        st.warning("Critical emotion detected. Redirecting to human agent.")
        with open(TRANSCRIPT_FILE, 'a') as f:
            f.write(f"Query: {transcript}\nResponse: Redirected to human agent due to {emotion} emotion.\n\n")
        return
    response_text = handle_emotion_based_response(emotion)
    response_audio = text_to_speech(response_text)
    response_path = os.path.join(RESPONSE_FOLDER, f'response_{emotion}.wav')
    with open(response_path, "wb") as f:
        f.write(response_audio)
    st.session_state.results.append({
        'file': file_path,
        'transcript': transcript,
        'emotion': emotion,
        'response_text': response_text,
        'response_audio': response_path
    })
    play_audio(response_audio)
    st.write(f"Recognized Text: {transcript}")
    st.write(f"Detected Emotion: {emotion}")
    st.write(f"Response: {response_text}")
    with open(TRANSCRIPT_FILE, 'a') as f:
        f.write(f"Query: {transcript}\nResponse: {response_text}\n\n")

def is_critical_emotion(emotion):
    return emotion in ['angry', 'fear', 'disgust']

def main():
    st.set_page_config(page_title="IVRS-SurveySparrow", page_icon="🐦")
    st.title("SER based IVRS")

    if 'results' not in st.session_state:
        st.session_state.results = []

    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav"], key="upload_audio")
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_audio.wav')
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        handle_audio(file_path)

    if st.button("Record Audio"):
        file_path = record_audio()
        handle_audio(file_path)

    if st.session_state.results:
        st.title("**Previous Interactions**")
        for result in st.session_state.results:
            st.write(f"**File Path:** {result['file']}")
            st.write(f"**Recognized Text:** {result['transcript']}")
            st.write(f"**Detected Emotion:** {result['emotion']}")
            st.write(f"**Response:** {result['response_text']}")
            st.audio(result['response_audio'], format='audio/wav')
            st.write("---")

if __name__ == "__main__":
    main()
