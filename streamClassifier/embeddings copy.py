import os
from PoseClassification.bootstrap_copy import BootstrapHelper
from datetime import datetime
import cv2


class StreamEmbedder:
    def __init__(
        self,
        stream_video: str,
        stream_image_out_dir: str,
        stream_csv_out_dir: str,
        output_video_path: str,
    ):
        self.stream_video = stream_video
        self.stream_image_out_dir = stream_image_out_dir
        self.stream_csv_out_dir = stream_csv_out_dir
        self.output_video_path = output_video_path
        self.bootstrap_helper = BootstrapHelper(
            images_out_folder=stream_image_out_dir,
            csvs_out_folder=stream_csv_out_dir,
        )

    def _get_frame_from_stream(self):
        """
        Returns the next frame from the video stream
        """
        cap = cv2.VideoCapture(self.stream_video)
        _, frame = cap.read()
        cap.release()
        return frame

    def _save_frame_to_directory(self, frame, directory):
        """
        Saves the given frame to the given directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jpg"
        filepath = os.path.join(directory, filename)
        cv2.imwrite(filepath, frame)
        
    def _remove_image_from_in_out_dir(self) -> None:
        target_dir_to_clean_in = os.path.join(self.stream_image_out_dir, "stream")
        self._clean_directory(directory=target_dir_to_clean_in)

    def generate_embbedings(self) -> None:
        """
        Returns the embeddings from the video stream
        """

        self.bootstrap_helper.bootstrap()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, 20.0, (640, 480))
        while True:
            frame = self._get_frame_from_stream()
            self._save_frame_to_directory(frame, os.path.join(self.stream_image_out_dir, "stream"))
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._save_frame_to_directory(frame, os.path.join(self.stream_image_out_dir, "stream"))
                break
        out.release()
        self._remove_image_from_in_out_dir()
        cv2.destroyAllWindows()
