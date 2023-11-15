import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse
import time
import mediapipe as mp
import os
import joblib

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
            if image_name.endswith(".jpg") or image_name.endswith(".png"):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (64, 64))
                yoga_postures.append(image)
                yoga_posture_labels.append(posture_name)

    return yoga_postures, yoga_posture_labels

def train_svm_model(X_train, yoga_posture_labels):
    svm = SVC(kernel="linear", C=1.0, random_state=42)
    svm.fit(X_train, yoga_posture_labels)
    return svm

def classify_posture(frame, pose, svm, threshold=0.75):
        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the pose landmarks using Mediapipe
        results = pose.process(frame_gray)
        if results.pose_landmarks is not None:
            # Extract the pose landmarks and calculate the HOG features
            landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark])
            landmarks = landmarks.flatten()
            # Predict the posture using the SVM model
            scores = svm.decision_function([landmarks])
            if np.max(scores) >= threshold:
                posture_pred = svm.predict([landmarks])[0]
            else:
                posture_pred = None
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

def yoga_classifier(input_path, display=False, output_path=None, save_model_path=None):
    # Load the yoga postures and their labels
    posture_dir = "./assets/images/train"
    yoga_postures, yoga_posture_labels = load_yoga_postures(posture_dir)

    if save_model_path is not None and os.path.exists(save_model_path):
        # Load the saved SVM model
        svm = joblib.load(save_model_path)
    else:
        # Extract features from the images using mediapipe
        X_train = []
        for posture in yoga_postures:
            results = mp_pose.Pose().process(posture)
            if results.pose_landmarks is not None:
                landmarks = np.array([[lmk.x, lmk.y] for lmk in results.pose_landmarks.landmark])
                landmarks = landmarks.flatten()
                X_train.append(landmarks)
        X_train = np.array(X_train)
        # Train a SVM model on the extracted features
        svm = train_svm_model(X_train, yoga_posture_labels)
        if save_model_path is not None:
            # Save the trained SVM model
            joblib.dump(svm, save_model_path)

    # Initialize Mediapipe pose detection
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Use the trained model to classify new images of the yoga postures
    if input_path == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
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
        if display:
            display = display_posture(frame, pose, posture_pred)
            if not display:
                break
        if output_path is not None:
            save_video(frame, out)
    cap.release()
    if output_path is not None:
        out.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yoga posture classifier')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or "camera" for webcam')
    parser.add_argument('--display', action='store_true', help='Display the video stream with the predicted posture')
    parser.add_argument('--output_path', type=str, help='Path to output video file')
    parser.add_argument('--save_model_path', type=str, help='Path to save the trained SVM model')
    args = parser.parse_args()

    yoga_classifier(args.input, args.display, args.output_path, args.save_model_path)
