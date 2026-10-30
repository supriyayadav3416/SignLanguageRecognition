# Sign Language Recognition System

A real-time Sign Language Recognition system that recognizes hand gestures representing English alphabets (A–Z) using a webcam.

## Features
- Real-time hand gesture detection using a webcam
- Recognition of English alphabets from A to Z
- Hand landmark-based feature extraction
- Machine Learning model for gesture classification

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas

## Machine Learning Algorithm
Random Forest Classifier is used to classify hand gestures based on extracted hand landmark features.

## Project Structure
- `app.py` – Runs the real-time recognition application.
- `collect_data.py` – Collects hand gesture data using a webcam.
- `train_model.py` – Trains the machine learning model.
- `sign_model.pkl` – Saved trained model.
- `hand_landmarker.task` – Hand landmark detection model.
- `data/sign_data.csv` – Collected hand gesture dataset.
- `requirements.txt` – Required Python libraries.

## Model Performance
- Classes: A–Z (26 alphabets)
- Training samples: 1,710
- Reported model accuracy: 99.71%

## How to Run

1. Clone this repository.
2. Install the required libraries:

   `pip install -r requirements.txt`

3. Run the application:

   `python app.py`

4. Allow webcam access and show hand gestures to the camera.

## Application
This project demonstrates how machine learning and computer vision can be used to recognize hand gestures and support sign language learning.