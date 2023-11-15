import os
from streamClassifier.embeddings import StreamEmbedder



# Variables
cwd = os.getcwd()
image_in_dir = os.path.join(cwd, "assets", "images", "stream_in")
image_out_dir = os.path.join(cwd, "assets", "images", "stream_out")
csv_out_dir = os.path.join(cwd, "assets", "stream", "csv")

stream_embedder = StreamEmbedder(
    stream_image_in_dir=image_in_dir,
    stream_image_out_dir=image_out_dir,
    stream_csv_out_dir=csv_out_dir,
)
