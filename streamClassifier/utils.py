import os
import pandas as pd

from streamClassifier.config import num_to_class_dict, class_to_num_dict


def build_header(num_embeddings):
    headers = (
        ["filename"]
        + [f"embedding_{i}" for i in range(1, num_embeddings + 1)]
        + ["class"]
    )
    return headers


def build_dataframe(source_dir: str) -> pd.DataFrame:
    filenames = os.listdir(source_dir)
    headers = build_header(99)
    df = pd.DataFrame(columns=headers)
    for file_ in filenames:
        classname = file_.split(".")[0]

        if classname not in class_to_num_dict.keys():
            classname = "unknown"

        filepath = os.path.join(source_dir, file_)
        df_temp = pd.read_csv(filepath, header=None)
        df_temp["class"] = classname
        df_temp.columns = headers
        df = pd.concat([df, df_temp], axis=0, ignore_index=True)

    df["class_num"] = df["class"].map(class_to_num_dict)
    return df
