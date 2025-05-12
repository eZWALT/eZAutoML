import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def prepare_data(df, target_column, scale=True, task_type="classification"):
    """
    Preprocess the given DataFrame: missing value handling, categorical encoding,
    target encoding (for classification), and optional scaling.

    Parameters:
    - df: Input DataFrame.
    - target_column: Name of the target column.
    - scale: Whether to scale numeric features. Default is True.
    - task_type: "classification" or "regression".

    Returns:
    - X: Preprocessed feature matrix.
    - y: Processed target array.
    - target_encoder: LabelEncoder for classification tasks (None otherwise).
    """
    
    # 1. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Drop non-predictive columns
    X = X.loc[:, ~X.columns.str.contains("id|name|url", case=False)]

    # 3. Impute missing values
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    for col in num_cols:
        if X[col].isnull().any():
            X[col] = SimpleImputer(strategy="mean").fit_transform(X[[col]]).ravel()

    for col in cat_cols:
        if X[col].isnull().any():
            X[col] = SimpleImputer(strategy="most_frequent").fit_transform(X[[col]]).ravel()

    # 4. Encode categorical features
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    # 5. Feature scaling
    if scale:
        X[num_cols] = StandardScaler().fit_transform(X[num_cols])

    # 6. Encode target (if classification)
    target_encoder = None
    if task_type == "classification":
        if not pd.api.types.is_integer_dtype(y) or y.nunique() != len(np.unique(y)):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        elif y.min() < 0:
            # e.g., from [-1, 1] to [0, 1]
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

    return X, y, target_encoder