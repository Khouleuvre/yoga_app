import os
from streamClassifier.embeddings import StreamEmbedder
from streamClassifier.classifier import SvcClassifier

# Variables
cwd = os.getcwd()
image_in_dir = os.path.join(cwd, "assets", "images", "stream_in")
image_out_dir = os.path.join(cwd, "assets", "images", "stream_out")
csv_out_dir = os.path.join(cwd, "assets", "csv", "stream")
training_csv_dir = os.path.join(cwd, "assets", "csv", "training")

stream_embedder = StreamEmbedder(
    stream_image_in_dir=image_in_dir,
    stream_image_out_dir=image_out_dir,
    stream_csv_out_dir=csv_out_dir,
)

# stream_embedder.generate_embbedings()
# svc_classifier = SvcClassifier(
#     training_csv_dir=training_csv_dir,
#     stream_csv_dir=csv_out_dir,
# )
# svc_classifier.fit(show_value=False)

# Fin initialisation

stream_embedder.generate_embbedings()
# svc_classifier.predict()
# report  = svc_classifier.get_precision_infos()

# print(type(report))

