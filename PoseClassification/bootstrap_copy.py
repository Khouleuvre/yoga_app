import cv2
from matplotlib import pyplot as plt
import numpy as np
import os, csv
from PIL import Image, ImageDraw
import sys
import tqdm
import shutil

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class BootstrapHelper(object):
    """Helps to bootstrap images and filter pose samples for classification."""

    def __init__(self, images_in_folder, images_out_folder, csvs_out_folder):
        self._images_in_folder = images_in_folder
        self._images_out_folder = images_out_folder
        self._csvs_out_folder = csvs_out_folder

        # Get list of pose classes and print image statistics.
        self._pose_class_names = sorted(
            [n for n in os.listdir(self._images_in_folder) if not n.startswith(".")]
        )

    def bootstrap(self, per_pose_class_limit=None):
        """Bootstraps images in a given folder.

        Required image in folder (same use for image out folder):
            pushups_up/
            image_001.jpg
            image_002.jpg
            ...
            pushups_down/
            image_001.jpg
            image_002.jpg
            ...
            ...

        Produced CSVs out folder:
            pushups_up.csv
            pushups_down.csv

        Produced CSV structure with pose 3D landmarks:
            sample_00001,x1,y1,z1,x2,y2,z2,....
            sample_00002,x1,y1,z1,x2,y2,z2,....
        """
        # Create output folder for CVSs.
        if not os.path.exists(self._csvs_out_folder):
            os.makedirs(self._csvs_out_folder)

        for pose_class_name in self._pose_class_names:
            print("Bootstrapping ", pose_class_name, file=sys.stderr)

            # Paths for the pose class.
            images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
            images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
            csv_out_path = os.path.join(self._csvs_out_folder, pose_class_name + ".csv")
            if not os.path.exists(images_out_folder):
                os.makedirs(images_out_folder)

            with open(csv_out_path, "w") as csv_out_file:
                csv_out_writer = csv.writer(
                    csv_out_file, delimiter=",", quoting=csv.QUOTE_MINIMAL
                )
                # Get list of images.
                image_files = [
                    os.path.join(images_in_folder, f)
                    for f in os.listdir(images_in_folder)
                    if f.endswith(".jpg")
                ]
                if per_pose_class_limit is not None:
                    image_files = image_files[:per_pose_class_limit]

                # Process each image.
                for image_file in tqdm.tqdm(image_files):
                    # Read image.
                    image = cv2.imread(image_file)
                    image_height, image_width, _ = image.shape

                    # Detect pose.
                    with mp_pose.Pose(
                        static_image_mode=True,
                        model_complexity=2,
                        min_detection_confidence=0.5,
                    ) as pose:
                        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                        # Skip if pose not detected.
                        if not results.pose_landmarks:
                            continue

                        # Write pose landmarks to CSV.
                        csv_out_writer.writerow(
                            [
                                os.path.basename(image_file),
                                *[
                                    f"{lmk.x * image_width},{lmk.y * image_height},{lmk.z}"
                                    for lmk in results.pose_landmarks.landmark
                                ],
                            ]
                        )

                    # Save image with pose landmarks.
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                    cv2.imwrite(
                        os.path.join(
                            images_out_folder, os.path.basename(image_file)
                        ),
                        image,
                    )

    def _clean_directory(self, directory: str) -> None:
        """
        Removes all files in the given directory
        """
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    
    def _is_unique_image_in_dir(self):
        """
        Returns true if the image is in the directory and there is only one (the last one to treat)
        """

        _unique_folder = os.listdir(self._images_in_folder)[0]
        _unique_folder_path = os.path.join(self._images_in_folder, _unique_folder)
        files = os.listdir(_unique_folder_path)

        if len(files) == 1:
            # print("There are images in the directory")
            return True
        else:
            # print("There are no images in the directory")
            return False
    