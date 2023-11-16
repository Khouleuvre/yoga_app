import argparse
from streamClassifier.classifier import SvcClassifier
from streamClassifier. import PoseClassifier
from streamClassifier.embeddings_copy import StreamEmbedder, PoseClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yoga posture classifier')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input video file or "camera" for webcam')
    parser.add_argument('--display', action='store_true', help='Display the video stream with the predicted posture')
    parser.add_argument('--output_path', type=str, help='Path to output video file')
    parser.add_argument('--save_model_path', type=str, help='Path to save the trained SVM model')
    args = parser.parse_args()

    # Initialize the classes
    svc_classifier = SvcClassifier(training_csv_dir, stream_csv_dir)
    pose_classifier = PoseClassifier(stream_csv_dir)
    stream_embedder = StreamEmbedder(args.input, stream_image_out_dir, stream_csv_out_dir, args.output_path)

    # Use the classes
    svc_classifier.fit()
    pose_classifier.fit()
    stream_embedder.generate_embbedings()
