# IVRS-SurveySparrow

## Overview
IVRS (Interactive Voice Response System) is designed to analyze customer emotions from audio inputs in real-time. The application uses speech-to-text, emotion recognition, and text-to-speech technologies to provide empathetic responses and escalate issues effectively.

## Features
- **Upload and Record Audio**: Users can upload or record audio files for analysis.
- **Emotion Recognition**: Identifies the emotion in the audio and provides a tailored response.
- **Speech-to-Text Conversion**: Converts speech in audio files to text.
- **Text-to-Speech Response**: Generates an audio response based on the detected emotion.
- **Previous Interactions Log**: Displays a history of previous interactions with recognized text, detected emotion, and responses.

## Tech Stack
- **Python**: Programming language.
- **Streamlit**: Web framework for interactive applications.
- **Librosa**: For audio processing and feature extraction.
- **SpeechRecognition**: For converting speech to text.
- **Pyttsx3**: For text-to-speech conversion.
- **Keras**: For loading the trained emotion recognition model.
- **Scikit-learn**: For feature scaling and encoding.
- **Sounddevice**: For recording audio.
- **Numpy**: For numerical operations.

## Installation

### Prerequisites
Ensure you have Python 3.7+ installed on your system.

### Install Dependencies
Create a virtual environment and install the required libraries using the `requirements.txt` file:
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```
## Training the Model

## Dataset

To train the model, you need the RAVDESS Emotional Speech Audio dataset. Follow these steps to download and prepare the dataset:

1. Download the Dataset

   Download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).

2. Prepare the Dataset

   - Unzip the downloaded file.
   - Place the extracted data into the `Data` folder in the root directory of the project.

   Your directory structure should look like this:
   - /Data
      - Datasets(audio files of 24 actors)
   - train_model.py
   - app.py

3. Proceed with the Training
After placing the dataset in the `Data` folder, follow the instructions in the project to preprocess and train the model.


## Train the Model  
```
python train_model.py
```
## Running the Application
**Start the Streamlit Application**: Run the following command to start the Streamlit server:
   ```
   streamlit run app.py
   ```
## Contributing
Feel free to contribute to this project by opening issues or creating pull requests. Your feedback and suggestions are welcome!
