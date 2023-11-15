import pandas as pd
import os

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from streamClassifier.utils import build_dataframe


class SvcClassifier:
    def __init__(self, training_csv_dir: str, stream_csv_dir:str):
        self.model = None
        self._training_dataset = self._get_training_dataset(training_csv_dir)

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
            "num_class_0": len(self._training_dataset[self._training_dataset["class_num"] == 0]),
            "num_class_1": len(self._training_dataset[self._training_dataset["class_num"] == 1]),
            "num_class_2": len(self._training_dataset[self._training_dataset["class_num"] == 2]),
            "num_class_3": len(self._training_dataset[self._training_dataset["class_num"] == 3]),
            "num_class_4": len(self._training_dataset[self._training_dataset["class_num"] == 4]),
        }
        return infos
    
    def get_precision_infos(self):
        print("Confusion Matrix:")
        print(self._confusion_matrix)
        print("Classification Report:")
        print(self._classification_report)
    
    def fit(self,  show_value: bool = True):
        """
        Evaluates the model
        """
        # Assuming X is your feature matrix and y is the target vector
        X, y = self._prepare_X_y_set()
        X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.3, random_state=42
        )

        svc = SVC(kernel="linear")  # You can change the kernel based on your data characteristics
        svc.fit(X_train, y_train)
        y_pred = svc.predict(X_test)

        self._confusion_matrix = confusion_matrix(y_test, y_pred)
        self._classification_report = classification_report(y_test, y_pred)
        self.model = svc
        
        if show_value:
            print("Confusion Matrix:")
            print(self._confusion_matrix)
            print("Classification Report:")
            print(self._classification_report)

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predicts the class of the input
        """
        if self.model is None:
            raise Exception("Model not found. Run fit() first.")
        y_pred = self.model.predict(X)
        return y_pred