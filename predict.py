import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from train_models import MIN_YEAR, MAX_YEAR, MODEL_CLASSES, PIPELINE_ALIASES, base_dir, SELECTED_FEATURES
from train_models import get_default_hyperparams, get_feature_names


# Global verbose flag
verbose: bool = False
vprint = print if verbose else (lambda *args, **kwargs: None)


def load_single_split(dataset_dir: str, year: int) -> Dict[str, Any]:
    """
    Loads the LOSO train and test split for a given target year.

    Parameters:
        dataset_dir (str): Path to the dataset root folder
                           (e.g. 'datasets/selectedStats_from1980').
        year (int): Target LOSO split year (e.g. 2026).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "X_train", "y_top1_train", "X_test", "y_top1_test"
            - "Name_test" (list of player names for the test split).
    """
    year_dir: str = os.path.join(dataset_dir, str(year))
    data: Dict[str, Any] = {}

    for split in ["train", "test"]:
        split_dir: str = os.path.join(year_dir, split)
        npz_path: str = os.path.join(split_dir, f"{split}.npz")
        name_path: str = os.path.join(split_dir, "Name.csv")

        npz_data = np.load(npz_path)
        data[f"X_{split}"] = npz_data["X"]
        data[f"y_top1_{split}"] = npz_data["y_top1"]

        if split == "test":
            data["Name_test"] = pd.read_csv(name_path)["Name"].tolist()

    return data


def predict_mvp_and_top_k(model: Any,
                          X_test: np.ndarray,
                          player_names: List[str],
                          top_k: int = 10) -> Dict[str, Any]:
    """
    Computes the predicted MVP and the top-k ranking by model probability.

    Parameters:
        model (Any): Trained classification model exposing a predict_proba method.
        X_test (np.ndarray): Feature matrix for the test split.
        player_names (List[str]): List of player names aligned with X_test.
        top_k (int): Number of top players to return.

    Returns:
        Dict[str, Any]: Dictionary with:
            - "predicted_mvp" (str): Name of the predicted MVP.
            - "predicted_mvp_prob" (float): Probability of the predicted MVP.
            - "predicted_top_k" (List[Dict[str, Any]]): List of top-k players with rank and prob.
    """
    probs: np.ndarray = model.predict_proba(X_test)[:, 1]

    # Sort players by predicted probability (descending)
    sorted_indices: np.ndarray = np.argsort(probs)[::-1]
    sorted_names: List[str] = [player_names[i] for i in sorted_indices]
    sorted_probs: np.ndarray = probs[sorted_indices]

    top_k = min(top_k, len(sorted_names))

    predicted_top_k: List[Dict[str, Any]] = []
    for rank_idx in range(top_k):
        predicted_top_k.append(
            {
                "rank": rank_idx + 1,
                "player": sorted_names[rank_idx],
                "prob": float(sorted_probs[rank_idx]),
            }
        )

    predicted_mvp: str = predicted_top_k[0]["player"]
    predicted_mvp_prob: float = predicted_top_k[0]["prob"]

    return {
        "predicted_mvp": predicted_mvp,
        "predicted_mvp_prob": predicted_mvp_prob,
        "predicted_top_k": predicted_top_k,
    }


def train_on_single_split_and_predict(dataset_dir: str,
                                      pipeline_name: str,
                                      year: int,
                                      model_key: str) -> Dict[str, Any]:
    """
    Trains a model on a single LOSO split and returns the predicted MVP and top-k players.

    Parameters:
        dataset_dir (str): Path to the dataset root folder for the selected pipeline.
        pipeline_name (str): Internal pipeline name (e.g. "allStats_from1980").
        year (int): Target LOSO split year (train on all other years, test on this one).
        model_key (str): Key of the model in MODEL_CLASSES (e.g. "logreg", "rf").

    Returns:
        Dict[str, Any]: Dictionary with predicted MVP and top-k ranking.
    """
    print()
    print(f"[INFO] Loading LOSO split for year {year}...")
    data: Dict[str, Any] = load_single_split(dataset_dir, year)

    X_train: np.ndarray = data["X_train"]
    y_train: np.ndarray = data["y_top1_train"]
    X_test: np.ndarray = data["X_test"]
    y_test: np.ndarray = data["y_top1_test"]  # not used, but loaded for completeness
    player_names_test: List[str] = data["Name_test"]

    # Optional feature selection
    selected_feature_names: Optional[List[str]] = None
    selected_feature_names = SELECTED_FEATURES.get(model_key, {}).get(pipeline_name, None)
    if selected_feature_names is None:
        print(f"[WARN] No selected features found for model '{model_key}' and pipeline "f"'{pipeline_name}', using full feature set.")

    if selected_feature_names is not None:
        vprint(f"[INFO] Using feature subset: {selected_feature_names}")
        all_feature_names: List[str] = get_feature_names(pipeline_name, year)
        selected_indices: List[int] = [
            i for i, name in enumerate(all_feature_names) if name in selected_feature_names
        ]
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]

    # Model setup
    if model_key not in MODEL_CLASSES:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(MODEL_CLASSES.keys())}")

    model_class = MODEL_CLASSES[model_key]
    model_name: str = model_class.__name__

    fixed_params: Dict[str, Any] = get_default_hyperparams(model_class, pipeline_name)
    print()
    print(f"[INFO] Training model '{model_name}' on pipeline '{pipeline_name}' for year {year}...")
    print(f"[INFO] Using hyperparameters: {fixed_params}")

    model = model_class(**fixed_params)
    model.fit(X_train, y_train)

    # Save model checkpoint
    models_base_dir: str = os.path.join(base_dir, "models_single_split")
    output_model_dir: str = os.path.join(models_base_dir, f"{model_name}_{pipeline_name}")
    os.makedirs(output_model_dir, exist_ok=True)

    checkpoint_path: str = os.path.join(output_model_dir, f"{model_name}_{year}.joblib")
    joblib.dump(model, checkpoint_path)
    print(f"[DONE] Model saved to {checkpoint_path}")

    # Predict MVP and top-10
    results: Dict[str, Any] = predict_mvp_and_top_k(
        model=model,
        X_test=X_test,
        player_names=player_names_test,
        top_k=10,
    )

    # Pretty print
    print()
    print(
        f"[RESULT] Predicted MVP for year {year}: {results['predicted_mvp']} "
        f"(prob={results['predicted_mvp_prob']:.4f})"
    )
    print(f"[RESULT] Predicted top-10 for year {year}:")
    for entry in results["predicted_top_k"]:
        print(f"  #{entry['rank']:2d}  {entry['player']:30s}  prob={entry['prob']:.4f}")

    return results


def main(pipeline="all1980", model="logreg", year=2026, verbose=False):
    datasets_base_dir: str = os.path.join(base_dir, "datasets")

    # Verbose global switch
    vprint = print if verbose else (lambda *a, **kw: None)

    # Resolve pipeline key to folder name
    if pipeline in PIPELINE_ALIASES:
        pipeline_name_cli: str = pipeline
        pipeline_name_internal: str = PIPELINE_ALIASES[pipeline]
    else:
        print(
            f"[ERROR] Unknown pipeline key '{pipeline}'. "
            f"Available: {list(PIPELINE_ALIASES.keys())}"
        )
        sys.exit(1)

    year: int = year
    if year < MIN_YEAR or year > MAX_YEAR:
        print(
            f"[ERROR] Year {year} is outside allowed range "
            f"[{MIN_YEAR}, {MAX_YEAR}]."
        )
        sys.exit(1)

    # For pipelines starting in 1980, enforce year >= 1980
    pipeline_min_year: int = 1956 if "1956" in pipeline_name_internal else 1980
    if year < pipeline_min_year:
        print(
            f"[ERROR] Year {year} is not valid for pipeline '{pipeline_name_internal}'. "
            f"Minimum year is {pipeline_min_year}."
        )
        sys.exit(1)

    dataset_dir: str = os.path.join(datasets_base_dir, pipeline_name_internal)

    train_on_single_split_and_predict(
        dataset_dir=dataset_dir,
        pipeline_name=pipeline_name_internal,
        year=year,
        model_key=model,
    )