import glob
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import math
import json
import matplotlib.pyplot as plt


from scipy.ndimage import zoom
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from skimage.transform import resize

import torch 
from torch.utils import data
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms
from medmnist import (
    INFO,
    PathMNIST,
    ChestMNIST,
    OCTMNIST,
    PneumoniaMNIST,
    BreastMNIST,
    BloodMNIST,
    TissueMNIST,
    OrganAMNIST,
    OrganCMNIST,
    OrganSMNIST,
)
from medmnist import (
    OrganMNIST3D,
    NoduleMNIST3D,
    AdrenalMNIST3D,
    FractureMNIST3D,
    VesselMNIST3D,
    SynapseMNIST3D,
)

# ===----------------------------------------------------------------------===#
# Datasets Loader                                                             #
#                                                                             #
# Basic class for evaluating all models through scikit-learn API              #
# abstracts the complexity of combining datasets -- models -- metrics         #
#                                                                             #
# Author: Walter J.T.V                                                        #
# ===----------------------------------------------------------------------===#


class DatasetLoader:
    def __init__(self, local_path: str = "data", metadata_path: str = "./data/metadata.json"):
        self.local_path = local_path
        self.dataset_groups = {
            "builtin": ["breast_cancer"],
            "local": None,  # Discovered dynamically
            "medmnist": [
                "pathmnist",
                #"chestmnist",
                "octmnist",
                "pneumoniamnist",
                "breastmnist",
                "bloodmnist",
                "tissuemnist",
                "organamnist",
                "organcmnist",
                "organsmnist",
            ],
            "medmnist3d": [
                "organmnist3d",
                "nodulmnist3d",
                "adrenalmnist3d",
                "fracturemnist3d",
                "vesselmnist3d",
                "synapsemnist3d",
            ],
        }
        self.no_scale_datasets = {
        "mnist",
        "fashion_mnist",
        "cifar10",
        "pathmnist",
        "chestmnist",
        "octmnist",
        "pneumoniamnist",
        "breastmnist",
        "bloodmnist",
        "tissuemnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
        "organmnist3d",
        "nodulmnist3d",
        "adrenalmnist3d",
        "fracturemnist3d",
        "vesselmnist3d",
        "synapsemnist3d",
        }
        self.datasets = {}  # Populated via method
        logger.info("DatasetLoader initialized.")
        self.metadata_path = metadata_path

    def _smart_read_csv(self, file):
        try:
            # Read the first 1024 characters to detect the delimiter and decimal format
            with open(file, "r", encoding="utf-8") as f:
                sample = f.read(1024)

            # Detect delimiter: semicolon or comma
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
            
            # Detect decimal: comma or dot
            decimal = "," if "," in sample and "." not in sample else "."

            # Read the CSV while ensuring that no column is treated as the index
            df = pd.read_csv(file, delimiter=delimiter, decimal=decimal, index_col=False)
            
            return df
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            return None


    def _load_local_datasets(self):
        datasets = {}
        logger.info("Searching for local CSV datasets...")
        csv_files = glob.glob(os.path.join(self.local_path, "*.csv"))

        with open(self.metadata_path, "r") as f:
            target_columns = json.load(f)

        self.dataset_groups["local"] = [os.path.basename(f) for f in csv_files]

        for file in csv_files:
            filename = os.path.basename(file)
            try:
                df = self._smart_read_csv(file)
                if df is None:
                    raise EnvironmentError("Could not read CSV.")

                if filename not in target_columns:
                    raise ValueError(f"Target column for {filename} not found in metadata!")

                target_col = target_columns[filename]
                X = df.drop(columns=[target_col])
                y = df[target_col]

                X, y = self._clean_and_preprocess_local(X, y)
                datasets[filename] = (X, y)
                
            except Exception as e:
                logger.warning(f"Skipping {file}: {repr(e)}")

        logger.success(f"Loaded {len(datasets)} local dataset(s).")
        return datasets

    def _convert_categoricals(self, X, y):
        X = X.copy()
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == object or isinstance(y[0], str):
            y = LabelEncoder().fit_transform(y)
        return X, y

    def _clean_and_preprocess_local(self, X, y):
        for col in X.columns:
            if "id" in col.lower() or "name" in col.lower() or "url" in col.lower():
                X = X.drop(col, axis=1)

        # Remove rows where y is not finite or NA
        if pd.api.types.is_float_dtype(y):
            mask = y.notna() & np.isfinite(y)
            y = y.loc[mask]
            y = y.astype(int)
            X = X.loc[mask]  # Align X with filtered y

        num_cols = X.select_dtypes(include=["number"]).columns
        cat_cols = X.select_dtypes(include=["object"]).columns

        for col in num_cols:
            if X[col].isnull().any():
                imputer = SimpleImputer(strategy="mean")
                X[col] = imputer.fit_transform(X[[col]]).ravel()

        for col in cat_cols:
            if X[col].isnull().any():
                imputer = SimpleImputer(strategy="most_frequent")
                X[col] = imputer.fit_transform(X[[col]]).ravel()
                
        return self._convert_categoricals(X, y)
    
    def _load_builtin_datasets(self):
        logger.info("Loading built-in datasets...")
        return {
            "breast_cancer": load_breast_cancer(return_X_y=True, as_frame=True),
        }

    def _load_torchvision_datasets(self):
        datasets = {}
        transform = transforms.Compose([transforms.ToTensor()])
        torchvision_datasets = {
            "fashion_mnist": FashionMNIST,
            "mnist": MNIST,
            "cifar10": CIFAR10,
        }

        logger.info("Loading torchvision datasets...")
        for name, dataset_cls in torchvision_datasets.items():
            try:
                dataset = dataset_cls(
                    root="data", train=True, download=True, transform=transform
                )
                X = torch.stack([img[0].flatten() for img in dataset]).numpy()
                y = (
                    dataset.targets.numpy()
                    if isinstance(dataset.targets, torch.Tensor)
                    else np.array(dataset.targets)
                )
                datasets[name] = (X, y)
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        return datasets

    def _load_medmnist(self, name):
        try:
            dataset_cls = {
                "pathmnist": PathMNIST,
                "chestmnist": ChestMNIST,
                "octmnist": OCTMNIST,
                "pneumoniamnist": PneumoniaMNIST,
                "breastmnist": BreastMNIST,
                "bloodmnist": BloodMNIST,
                "tissuemnist": TissueMNIST,
                "organamnist": OrganAMNIST,
                "organcmnist": OrganCMNIST,
                "organsmnist": OrganSMNIST,
            }.get(name.lower())

            if dataset_cls is None:
                raise ValueError(f"{name} not recognized.")

            dataset = dataset_cls(root="data", split="train", download=True)

            # Check if `targets` or `labels` is available
            if hasattr(dataset, 'targets'):
                y = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
            elif hasattr(dataset, 'labels'):
                y = dataset.labels.numpy() if isinstance(dataset.labels, torch.Tensor) else np.array(dataset.labels)
            else:
                raise AttributeError(f"Dataset '{name}' has no 'targets' or 'labels' attribute.")

            X = torch.stack([transforms.ToTensor()(img[0]).flatten() for img in dataset]).numpy()

            return X, y

        except Exception as e:
            logger.warning(f"Error loading {name}: {e}")
            return None, None

    def _load_medmnist3d(self, name):
        try:
            dataset_cls = {
                "organmnist3d": OrganMNIST3D,
                "nodulmnist3d": NoduleMNIST3D,
                "adrenalmnist3d": AdrenalMNIST3D,
                "fracturemnist3d": FractureMNIST3D,
                "vesselmnist3d": VesselMNIST3D,
                "synapsemnist3d": SynapseMNIST3D,
            }.get(name.lower())

            if dataset_cls is None:
                raise ValueError(f"{name} not recognized.")

            dataset = dataset_cls(root="data", split="train", download=True)
            loader = data.DataLoader(dataset, batch_size=128, shuffle=False)

            all_X, all_y = [], []
            for batch_X, batch_y in loader:
                all_X.append(batch_X.view(batch_X.size(0), -1))
                all_y.append(batch_y)

            X = torch.cat(all_X).numpy()
            y = torch.cat(all_y).squeeze().numpy()
            return X, y
        except Exception as e:
            logger.warning(f"Error loading {name}: {e}")
            return None, None

    def _preprocess_data(self, X, y, test_size=0.2, scale=True):
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        return X, y

    def load_selected_datasets(self, groups=None, names=None):
        selected = {}

        if groups:
            logger.info(f"Loading datasets from groups: {groups}")
            for group in groups:
                if group == "local":
                    selected.update(self._load_local_datasets())
                elif group == "builtin":
                    selected.update(self._load_builtin_datasets())
                elif group == "torchvision":
                    selected.update(self._load_torchvision_datasets())
                elif group == "medmnist":
                    for name in self.dataset_groups["medmnist"]:
                        X, y = self._load_medmnist(name)
                        if X is not None:
                            selected[name] = (X, y)
                elif group == "medmnist3d":
                    for name in self.dataset_groups["medmnist3d"]:
                        X, y = self._load_medmnist3d(name)
                        if X is not None:
                            selected[name] = (X, y)
                else:
                    logger.warning(f"Unknown group: {group}")

        if names:
            logger.info(f"Loading datasets by name: {names}")
            for name in names:
                if name in selected:
                    continue  # Skip if already loaded

                if name in self.dataset_groups["medmnist"]:
                    X, y = self._load_medmnist(name)
                elif name in self.dataset_groups["medmnist3d"]:
                    X, y = self._load_medmnist3d(name)
                elif name in ["mnist", "fashion_mnist", "cifar10"]:
                    X, y = self._load_torchvision_datasets().get(name, (None, None))
                elif name == "breast_cancer":
                    X, y = self._load_builtin_datasets()[name]
                else:
                    local_datasets = self._load_local_datasets()
                    X, y = local_datasets.get(name, (None, None))

                if X is not None and y is not None:
                    selected[name] = (X, y)

        logger.info("Preprocessing all selected datasets...")
        processed = {}
        # Do not scale image-based datasets :)
        for name, (X, y) in selected.items():
            try:
                processed[name] = self._preprocess_data(X, y, scale=(name not in self.no_scale_datasets))
            except Exception as e:
                logger.error(f"Error preprocessing {name}: {e}")

        self.datasets = processed
        logger.success(f"Loaded and preprocessed {len(processed)} dataset(s).")
        return processed

    def get_datasets(self):
        return self.datasets
    
    def load_user_datasets(self, file_paths: list[str], metadata: dict[str, str]):
        """
        Load user-provided datasets from CSV files with minimal tabular ML treatment.
        
        :param file_paths: list of CSV file paths
        :param metadata: dict of filename -> target_column
        :return: dict of {filename: (X, y)}
        """
        user_datasets = {}

        for path in file_paths:
            filename = os.path.basename(path)
            try:
                if filename not in metadata:
                    raise ValueError(f"Missing target column info for: {filename}")

                df = self._smart_read_csv(path)
                if df is None:
                    raise ValueError(f"Failed to read CSV: {filename}")

                target_col = metadata[filename]
                X = df.drop(columns=[target_col])
                y = df[target_col]

                X, y = self._clean_and_preprocess_local(X, y)
                user_datasets[filename] = (X, y)

                logger.success(f"Loaded user dataset: {filename} | X: {X.shape} | y: {y.shape}")
            except Exception as e:
                logger.warning(f"Skipping {filename}: {repr(e)}")

        self.datasets.update(user_datasets)
        return user_datasets



def test_main_loading():
    loader = DatasetLoader(local_path="data")

    # Example usage:
    datasets = loader.load_selected_datasets(
        groups=[
         #"builtin",
         #"medmnist3d",
         #"medmnist",
         "local",
         #"torchvision"
        ]
    )

    for name, (X, Y) in datasets.items():
        logger.info(f"\n🗂️ Dataset: {name}")
        logger.info(f"  ➤ X: {X.shape}")
        logger.info(f"  ➤ Y: {Y.shape}")
        logger.info(f"  ➤ Y type: {type(Y)}")
        logger.info(f"  ➤ Y dtype: {Y.dtype}")

def test_user_custom_data():
    user_files = ["/home/wtroiani/lol1.csv", "/home/wtroiani/lol2.csv"]
    user_metadata = {
        "lol1.csv": "Target",
        "lol2.csv": "smoking",
    }

    loader = DatasetLoader(local_path="../../data", metadata_path="../../data/metadata.json")
    user_datasets = loader.load_user_datasets(user_files, user_metadata)

    for name, (X, y) in user_datasets.items():
        print(f"{name}: X={X.shape}, y={y.shape}, y type={y.dtype}")


if __name__ == "__main__":
    test_user_custom_data()
    