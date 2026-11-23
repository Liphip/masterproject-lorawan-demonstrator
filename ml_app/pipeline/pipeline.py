import datetime
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Sequence
import json

class CustomPipeline(Pipeline):
    def __init__(self, steps: Sequence[tuple], version: str, model_type: str, feature_order: list, created_at: "datetime.datetime | None" = None):
        super().__init__(steps)
        self.version = version
        self.model_type = model_type
        self.feature_order = feature_order
        # default evaluated per instance (was evaluated once at import time)
        self.created_at = created_at if created_at is not None else datetime.datetime.now()
    
    def display(self):
        """Display the pipeline steps and metadata."""
        print(f"Pipeline Version: {self.version}")
        print(f"Model Type: {self.model_type}")
        print(f"Feature Order: {self.feature_order}")
        print(f"Created At: {self.created_at.isoformat()}")
        print("Pipeline Steps:")
        for name, step in self.steps:
            print(f" - {name}: {step}")

    def get_metadata(self):
        """Get the metadata of the pipeline.
        Returns:
            dict: A dictionary containing the metadata of the pipeline.
        """
        return {
            "version": self.version,
            "model_type": self.model_type,
            "feature_order": self.feature_order,
            "created_at": self.created_at.isoformat()
        }

    def set_metadata(self, metadata: dict):
        """Set the metadata of the pipeline.
        Args:
            metadata (dict): A dictionary containing the metadata of the pipeline.
        """
        self.version = metadata.get("version", self.version)
        self.model_type = metadata.get("model_type", self.model_type)
        self.feature_order = metadata.get("feature_order", self.feature_order)
        created_at_str = metadata.get("created_at")
        if created_at_str:
            self.created_at = datetime.datetime.fromisoformat(created_at_str)

    def store(self, path: str):
        """Store the pipeline to a file.
        Uses joblib to store the pipeline to a file.
        Also creates a metadata file in the same directory.

        Args:
            path (str): The path to the file where the pipeline will be stored.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        metadata = self.get_metadata()
        metadata_path = os.path.join(path, f"{self.version}_metadata.json")
        model_path = os.path.join(path, f"{self.version}_model.joblib")
        joblib.dump(self, model_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    @staticmethod
    def load(path: str, version: str) -> "CustomPipeline":
        """Load the pipeline from a file.
        Uses joblib to load the pipeline from a file.
        Also loads the metadata from the corresponding metadata file.

        Args:
            path (str): The path to the file where the pipeline is stored.
            version (str): The version of the pipeline to load.

        Returns:
            CustomPipeline: The loaded pipeline.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")

        model_path = os.path.join(path, f"{version}_model.joblib")
        metadata_path = os.path.join(path, f"{version}_metadata.json")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        pipeline = joblib.load(model_path)
        pipeline.set_metadata(metadata)

        return pipeline
    
    def fit(self, X, y=None, **fit_params):
        """Fit the pipeline to the data.
        Overrides the fit method to include metadata update.

        Args:
            X: The input data.
            y: The target variable.
            fit_params: Additional fit parameters.

        Returns:
            self: The fitted pipeline.
        """
        X = X[self.feature_order] # type: ignore
        return super().fit(X, y, **fit_params)

    def predict(self, X, **predict_params):
        """Predict using the pipeline.
        Overrides the predict method to ensure the input data is in the correct feature order.

        Args:
            X: The input data.
            predict_params: Additional predict parameters.
        Returns:
            The predicted values.
        """
        X = X[self.feature_order] # type: ignore
        return super().predict(X, **predict_params)

    def predict_proba(self, X, **predict_params):
        """Predict probabilities using the pipeline.
        Overrides the predict_proba method to ensure the input data is in the correct feature order.

        Args:
            X: The input data.
            predict_params: Additional predict parameters.
        Returns:
            The predicted probabilities.
        """        
        X = X[self.feature_order] # type: ignore
        return super().predict_proba(X, **predict_params)