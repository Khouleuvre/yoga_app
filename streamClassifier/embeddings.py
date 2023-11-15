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
        self._is_image_in_dir()

    def _is_unique_image_in_dir(self):
        """
        Returns true if the image is in the directory
        """

        _unique_folder = os.listdir(self.stream_image_in_dir)[0]
        _unique_folder_path = os.path.join(self.stream_image_in_dir, _unique_folder)
        files = os.listdir(_unique_folder_path)

        if len(files) == 1:
            print("There are images in the directory")
            return True
        else:
            print("There are no images in the directory")
            return False

    def _remove_image_from_in_dir(self, image_name):
        pass

    def get_embeddings_from_stream(self, stream):
        """
        Returns the embeddings from the stream
        """

        is_unique_image_in_dir = self._is_image_in_dir()

        pass


def get_embeddings_from_image():
    """
    Returns the embeddings from the image
    """
    pass
