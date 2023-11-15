from streamClassifier.utils import build_dataframe


class SvcClassifier:
    def __init__(self, training_csv_dir: str):
        self.model = None
        self.training_dataset = self._get_training_dataset(training_csv_dir)

    def _get_training_dataset(self, training_csv_dir) -> pd.DataFrame:
        df = build_dataframe(source_dir=self.training_csv_dir)
        return df

    def _fit_svc(self, X, y):
        """
        Fits the SVC model
        """
        self.model = SVC(kernel="linear", C=0.025, random_state=101)
        self.model.fit(X, y)

    bootstrap_images_in_folder = os.path.join(cwd, "assets", "images", "all")
    bootstrap_images_out_folder = os.path.join(cwd, "assets", "images", "bootstraped")
    bootstrap_csvs_out_folder = os.path.join(cwd, "assets", "csv")
