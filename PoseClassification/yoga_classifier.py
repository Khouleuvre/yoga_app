import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse
import time
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def load_yoga_postures(posture_dir):
    yoga_postures = []
    yoga_posture_labels = []

    for posture_name in os.listdir(posture_dir):
        if posture_name.startswith("."):
            continue
        posture_folder = os.path.join(posture_dir, posture_name)
        for image_name in os.listdir(posture_folder):
            if image_name.startswith("."):
                continue
            image_path = os.path.join(posture_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            yoga_postures.append(image)
            yoga_posture_labels.append(posture_name)

    return yoga_postures, yoga_posture_labels

def train_svm_model(X_train, yoga_posture_labels):
    svm = SVC(kernel="linear", C=1.0, random_state=42)
    svm.fit(X_train, yoga_posture_labels)
    return svm

def classify_posture(frame, pose, svm):
    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the pose landmarks using Mediapipe
    results = pose.process(frame_gray)
    if results.pose_landmarks is not None:
        # Extract the pose landmarks and calculate the HOG features
        landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark])
        landmarks = landmarks.flatten()
        # Predict the posture using the SVM model
        posture_pred = svm.predict([landmarks])[0]
    else:
        posture_pred = None
    return posture_pred

def display_posture(frame, pose, posture_pred):
    # Draw the pose landmarks and the predicted posture on the frame
    mp_drawing.draw_landmarks(frame, pose.process(frame).pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.putText(frame, posture_pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    else:
        return True

def save_video(frame, out):
    out.write(frame)

def yoga_classifier():
    parser = argparse.ArgumentParser(description='Classify yoga postures in a video stream.')
    parser.add_argument('input', metavar='INPUT', type=str, help='path to input video file or "camera" for live video stream')
    parser.add_argument('--display', action='store_true', help='display the video stream with the recognized posture and a progress bar')
    parser.add_argument('--save', metavar='OUTPUT', type=str, help='path to output video file')
    args = parser.parse_args()

    # Load the yoga postures and their labels
    posture_dir = "assets/images/train"
    yoga_postures, yoga_posture_labels = load_yoga_postures(posture_dir)

    # Extract features from the images using mediapipe
    

    # Train a SVM model on the extracted features
    X_train = []
    for posture in yoga_postures:
        results = mp_pose.Pose().process(posture)
        if results.pose_landmarks is not None:
            landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark])
            landmarks = landmarks.flatten()
            X_train.append(landmarks)
    X_train = np.array(X_train)
    svm = train_svm_model(X_train, yoga_posture_labels)

    # Initialize Mediapipe pose detection
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Use the trained model to classify new images of the yoga postures
    if args.input == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.save, fourcc, 20.0, (640, 480))
    start_time = time.time()
    posture_timer = 0
    posture_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        posture_pred = classify_posture(frame, pose, svm)
        if posture_pred == "maintained":
            posture_timer = time.time() - start_time
            if posture_timer >= 30:
                # Flash the image
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)
                cv2.imshow('frame', np.zeros_like(frame))
                cv2.waitKey(1000)
                # Display progress line
                posture_count += 1
                progress = int((posture_timer / 30) * 100)
                cv2.putText(frame, f"Progress: {progress}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            start_time = time.time()
            posture_timer = 0
        if args.display:
            display = display_posture(frame, pose, posture_pred)
            if not display:
                break
        if args.save:
            save_video(frame, out)
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()
