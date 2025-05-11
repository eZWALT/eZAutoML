import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def prepare_data(df, target_column, scale=True):
    """
    Preprocess the given DataFrame, handling missing values, categorical encoding,
    and feature scaling.

    Parameters:
    - df: The input DataFrame.
    - target_column: The name of the column to predict.
    - scale: Whether or not to scale numeric features. Default is True.

    Returns:
    - X: The features as a NumPy array.
    - y: The target as a NumPy array.
    """
    
    # Step 1: Separate target variable from features
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Step 2: Clean the features by removing non-predictive columns (e.g., "id", "name")
    X = X.loc[:, ~X.columns.str.contains("id|name|url", case=False)]

    # Step 3: Handle missing values (imputation)
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Impute numeric features
    for col in num_cols:
        if X[col].isnull().any():
            imputer = SimpleImputer(strategy="mean")
            X[col] = imputer.fit_transform(X[[col]]).ravel()

    # Impute categorical features
    for col in cat_cols:
        if X[col].isnull().any():
            imputer = SimpleImputer(strategy="most_frequent")
            X[col] = imputer.fit_transform(X[[col]]).ravel()

    # Step 4: Convert categorical features to numeric using LabelEncoder
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Step 5: Scale the numeric features (Standardization)
    if scale:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

    # Step 6: Handle target variable
    if y.dtype == object or isinstance(y[0], str):
        y = LabelEncoder().fit_transform(y)
    
    # Step 7: Optionally split into train and test sets
    # You can add this if needed:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y
