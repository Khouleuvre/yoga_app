import os
import numpy as np

from mediapipe.python.solutions import pose as mp_pose
from PoseClassification.bootstrap import BootstrapHelper


class StreamEmbedder:
    def __init__(
        self,
        stream_image_in_dir: str,
        stream_image_out_dir: str,
        stream_csv_out_dir: str,
    ):
        self.stream_image_in_dir = stream_image_in_dir
        self.stream_image_out_dir = stream_image_out_dir
        self.stream_csv_out_dir = stream_csv_out_dir
        self.bootstrap_helper = BootstrapHelper(
            images_in_folder=stream_image_in_dir,
            images_out_folder=stream_image_out_dir,
            csvs_out_folder=stream_csv_out_dir,
        )

    # Trouver une autre façon de coder cette fonction
    def _is_unique_image_in_dir(self):
        """
        Returns true if the image is in the directory and there is only one (the last one to treat)
        """

        _unique_folder = os.listdir(self.stream_image_in_dir)[0]
        _unique_folder_path = os.path.join(self.stream_image_in_dir, _unique_folder)
        files = os.listdir(_unique_folder_path)

        if len(files) == 1:
            # print("There are images in the directory")
            return True
        else:
            # print("There are no images in the directory")
            return False

    def _clean_directory(self, directory: str) -> None:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    # Trouver une autre façon de coder cette fonction
    def _remove_image_from_in_out_dir(self) -> None:
        target_dir_to_clean_in = os.path.join(self.stream_image_in_dir, "stream")
        target_dir_to_clean_out = os.path.join(self.stream_image_out_dir, "stream")
        # self._clean_directory(directory=target_dir_to_clean_in)
        # self._clean_directory(directory=target_dir_to_clean_out)

    def _get_embeddings_from_stream(self):
        """
        Returns the embeddings from the stream
        """

        if self._is_unique_image_in_dir():
            print("There is ONE image only in the directory")
        else:
            return false

    def _get_landmarks_from_rgb(self, input_frame: np.ndarray = None):
        output_frame = input_frame.copy()

        with mp_pose.Pose() as pose_tracker:
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

        # Save landmarks if pose was detected.
        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = (
                output_frame.shape[0],
                output_frame.shape[1],
            )
            pose_landmarks = np.array(
                [
                    [
                        lmk.x * frame_width,
                        lmk.y * frame_height,
                        lmk.z * frame_width,
                    ]
                    for lmk in pose_landmarks.landmark
                ],
                dtype=np.float32,
            )
            assert pose_landmarks.shape == (
                33,
                3,
            ), "Unexpected landmarks shape: {# pose_landmarks_list = pose_landmarks.flatten().astype(str).tolist()}"

        # print(pose_landmarks)
        pose_landmarks = pose_landmarks.flatten().astype(str).tolist()
        return pose_landmarks

    def generate_embbedings(self, input_frame: np.ndarray = None) -> None:
        """
        Returns the embeddings from the stream
        """
        landmarks = self._get_landmarks_from_rgb(input_frame=input_frame)
        # if input_frame is None:
        #     input_frame = os.listdir(self.stream_image_in_dir)[0]
        #     print(f"image: {input_frame}")
        # else:
        #     pass
        # self.bootstrap_helper.get_embeddings(input_frame=input_frame)
        # self.bootstrap_helper.bootstrap()
        # self._remove_image_from_in_out_dir()
        return landmarks
