import pandas as pd
import numpy as np
import os


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from streamClassifier.utils import build_dataframe

from streamClassifier.config import num_to_class_dict, class_to_num_dict


class SvcClassifier:
    def __init__(self, training_csv_dir: str, stream_csv_dir: str):
        self.model = None
        self._training_dataset = self._get_training_dataset(training_csv_dir)
        self._stream_csv_dir = stream_csv_dir

    def _get_training_dataset(self, training_csv_dir) -> pd.DataFrame:
        df = build_dataframe(source_dir=training_csv_dir)
        return df

    def _prepare_X_y_set(self) -> tuple:
        df = self._training_dataset
        X = df.drop(
            ["filename", "class", "class_num"], axis=1
        )  # Assuming 'label' is the column with class names/numbers
        y = df["class_num"]
        return X, y

    def get_training_data_infos(self) -> dict:
        infos = {
            "number_of_samples": len(self._training_dataset),
            "num_class_0": len(
                self._training_dataset[self._training_dataset["class_num"] == 0]
            ),
            "num_class_1": len(
                self._training_dataset[self._training_dataset["class_num"] == 1]
            ),
            "num_class_2": len(
                self._training_dataset[self._training_dataset["class_num"] == 2]
            ),
            "num_class_3": len(
                self._training_dataset[self._training_dataset["class_num"] == 3]
            ),
            "num_class_4": len(
                self._training_dataset[self._training_dataset["class_num"] == 4]
            ),
        }
        return infos

    def get_precision_infos(self):
        print("Confusion Matrix:")
        print(self._confusion_matrix)
        print("Classification Report:")
        print(self._classification_report)

        return self._classification_report

    def _compute_confidence_interval(
        self,
        decision_values: np.array,
        confidence_threshold: float = 0.1,
    ):
        max_value = []
        delta = []

        for decision in decision_values:
            # Get indices of two largest elements
            indices = np.argsort(decision)[-2:]
            largest_values = decision[indices]
            delta.append(largest_values[1] - largest_values[0])
            max_value.append(np.max(decision))

        self.high_confidence_max = np.mean(max_value)  # * (1 + confidence_threshold)
        self.high_confidence_delta = np.mean(delta)  # * (1 + confidence_threshold)

    def fit(self, show_value: bool = True, confidence_threshold: float = 0.1):
        """
        Evaluates the model
        """
        # Assuming X is your feature matrix and y is the target vector
        X, y = self._prepare_X_y_set()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        svc = SVC(
            kernel="linear"
        )  # You can change the kernel based on your data characteristics
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        self._confusion_matrix = confusion_matrix(y_test, y_pred)
        self._classification_report = classification_report(y_test, y_pred)
        self.model = svc
        X = np.array(X_test)
        X = np.reshape(X, (X.shape[0], -1))
        decision_values = self.model.decision_function(X)
        self._compute_confidence_interval(
            decision_values=decision_values, confidence_threshold=confidence_threshold
        )

        if show_value:
            print("Confusion Matrix:")
            print(self._confusion_matrix)
            print("Classification Report:")
            print(self._classification_report)

    def predict(self) -> list:
        """
        Predicts the class of the input
        """
        if self.model is None:
            raise Exception("Model not found. Run fit() first.")

        df = build_dataframe(source_dir=self._stream_csv_dir)
        X = df.drop(["filename", "class", "class_num"], axis=1)

        y_pred = self.model.predict(X)

        classnum_pred = y_pred[0]
        classname_pred = num_to_class_dict[classnum_pred]

        print(f"Predicted class: {classname_pred}")
        print(f"Predicted class number: {classnum_pred}")

    def predict_stream(self, data: np.array) -> tuple:
        """
        Predicts the class of the input
        """
        if self.model is None:
            raise Exception("Model not found. Run fit() first.")

        y_pred = self.model.predict(data)

        classnum_pred = y_pred[0]
        classname_pred = num_to_class_dict[classnum_pred]
        decision_values = self.model.decision_function(data)

        max_value = np.max(decision_values)
        two_max_indices = np.argsort(decision_values)[0][-2:]

        delta = (
            decision_values[0][two_max_indices[1]]
            - decision_values[0][two_max_indices[0]]
        )

        max_criteria = max_value > 4.31  # self.high_confidence_max
        delta_criteria = delta < 1.02  # self.high_confidence_delta

        print(f"Criteria Max: {self.high_confidence_max}")
        print(f"Criteria Delta: {self.high_confidence_delta}")

        print(f"Max value: {max_value}")
        print(f"Delta: {delta}")

        if max_criteria and delta_criteria:
            return classname_pred
        else:
            return "unknown"
