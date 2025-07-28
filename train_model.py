# train_model.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas(desc="Processing Lines")

def compute_class_weights(y, power=0.4):
    class_counts = np.bincount(y)
    total_samples = len(y)
    num_classes = len(class_counts)
    weights = {
        i: (total_samples / (num_classes * class_counts[i])) ** power
        for i in range(num_classes) if class_counts[i] > 0
    }
    return weights

def train_model(df: pd.DataFrame, model_path: str):
    if df.empty:
        print("Error: The training dataframe is empty. Cannot train model.")
        return

    print("Preparing data for training...")
    df = df.copy()
    
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    # --- Feature Selection (Layout and Pattern features ONLY) ---
    feature_cols = [
        "font_size", "is_bold", "x", "y", "x_norm", "y_norm", "page", "is_title",
        "relative_font_size", "is_all_caps", "is_mostly_digits", "word_count", "char_count",
        "y_gap_from_prev", "x_diff_from_prev", "font_diff_from_prev",
        "ends_with_colon", "starts_with_numbering"
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Feature column '{col}' not found. Filling with 0.")
            df[col] = 0

    X = df[feature_cols]
    y = df["label_enc"]

    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        print(f"Error: The least populated class has only {min_class_count} member(s). Cannot train.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    print("Computing class weights for imbalanced data...")
    weights_map = compute_class_weights(y_train)
    sample_weight = y_train.progress_apply(lambda label: weights_map.get(label, 1))

    print("Initializing model with optimal hyperparameters...")
    best_params = {
        'colsample_bytree': 0.73, 'gamma': 0.2, 'learning_rate': 0.18,
        'max_depth': 7, 'min_child_weight': 4, 'n_estimators': 2000,
        'subsample': 0.9, 'tree_method': "hist", 'device': "cuda",
        'eval_metric': "mlogloss", 'early_stopping_rounds': 10
    }
    
    model = xgb.XGBClassifier(**best_params)

    print("\nStarting model training...")
    model.fit(
        X_train, y_train, 
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False # Set to True to see round-by-round progress
    )

    joblib.dump((model, le), model_path)
    print(f"\n[✓] Saved final model and label encoder to {model_path}")
