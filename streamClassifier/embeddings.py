import os
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

    def _remove_image_from_in_out_dir(self) -> None:
        target_dir_to_clean_in = os.path.join(self.stream_image_in_dir, "stream")
        target_dir_to_clean_out = os.path.join(self.stream_image_out_dir, "stream")
        self._clean_directory(directory=target_dir_to_clean_in)
        self._clean_directory(directory=target_dir_to_clean_out)

    def _get_embeddings_from_stream(self):
        """
        Returns the embeddings from the stream
        """

        if self._is_unique_image_in_dir():
            print("There is ONE image only in the directory")
        else:
            return false

    def generate_embbedings(self) -> None:
        """
        Returns the embeddings from the stream
        """

        self.bootstrap_helper.bootstrap()
        self._remove_image_from_in_out_dir()
