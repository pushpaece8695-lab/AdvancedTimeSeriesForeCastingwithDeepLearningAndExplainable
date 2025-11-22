"""
Advanced Multivariate Time Series Forecasting with LSTM and SHAP Explainability.

This script:
1. Programmatically generates a realistic 3-year daily multivariate time series dataset
   with at least 5 features.
2. Prepares the data for multi-step sequence forecasting.
3. Builds and trains an LSTM-based deep learning model in TensorFlow/Keras.
4. Performs a simple manual hyperparameter grid search.
5. Uses early stopping and learning-rate scheduling.
6. Evaluates the final model with RMSE, MAE, MAPE, and a custom business loss metric.
7. Applies SHAP to explain feature importance for several forecasted time steps.

The code is modular, uses docstrings (PEP-257 style), and is written to be
production-ready and easy to extend.
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

import shap


# ----------------------------------------------------------------------
# Reproducibility helpers
# ----------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------
# Configuration dataclasses
# ----------------------------------------------------------------------
@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    n_years: int = 3
    freq: str = "D"          # daily data
    n_features: int = 5      # number of input features (excluding target)
    target_name: str = "load"
    start_date: str = "2018-01-01"


@dataclass
class ModelConfig:
    """Configuration for sequence modeling."""
    input_window: int = 30   # length of input history (days)
    output_horizon: int = 7  # forecast horizon (days)
    max_epochs: int = 40
    patience: int = 5


@dataclass
class SearchSpace:
    """Hyperparameter search space for manual grid search."""
    units_list: List[int] = (32, 64)
    lr_list: List[float] = (1e-3, 5e-4)
    batch_sizes: List[int] = (32, 64)


# ----------------------------------------------------------------------
# Data generation and preprocessing
# ----------------------------------------------------------------------
def generate_synthetic_multivariate_ts(config: DataConfig) -> pd.DataFrame:
    """
    Programmatically generate a realistic multivariate time series dataset.

    Features:
        - load (target): base sinusoidal demand + trend + noise
        - temperature: yearly seasonality
        - is_weekend: 0/1 flag
        - promo: random promotion days impacting load
        - special_event: occasional spikes
        - day_of_week: encoded as integer

    Returns:
        DataFrame indexed by date with all features including the target.
    """
    date_index = pd.date_range(
        start=config.start_date,
        periods=config.n_years * 365,
        freq=config.freq,
    )

    n = len(date_index)
    t = np.arange(n)

    # Base seasonal pattern for load
    seasonal = 10 * np.sin(2 * np.pi * t / 365.0)
    trend = 0.01 * t
    noise = np.random.normal(scale=2.0, size=n)

    # Temperature (another sinusoid with different amplitude and phase)
    temperature = 20 + 10 * np.sin(2 * np.pi * (t + 30) / 365.0) + \
        np.random.normal(scale=1.0, size=n)

    # Weekend flag
    dow = date_index.dayofweek
    is_weekend = (dow >= 5).astype(int)

    # Promo days
    promo = np.zeros(n)
    promo_days = np.random.choice(n, size=int(0.1 * n), replace=False)
    promo[promo_days] = 1

    # Special events (e.g., outages, sports events)
    special_event = np.zeros(n)
    event_days = np.random.choice(n, size=int(0.03 * n), replace=False)
    special_event[event_days] = 1

    # Demand/load responds to these factors
    base_load = 50 + seasonal + trend + noise
    load = base_load + 3 * promo + 5 * special_event - 4 * is_weekend \
        - 0.5 * (temperature - 20)

    df = pd.DataFrame(
        {
            "date": date_index,
            "load": load,
            "temperature": temperature,
            "is_weekend": is_weekend,
            "promo": promo,
            "special_event": special_event,
            "day_of_week": dow,
        }
    )

    df.set_index("date", inplace=True)
    return df


def create_sequences(
    df: pd.DataFrame,
    model_cfg: ModelConfig,
    target_col: str = "load",
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Transform a time series DataFrame into input/target sequences.

    Args:
        df: DataFrame with features and target.
        model_cfg: ModelConfig with input_window and output_horizon.
        target_col: Name of the target column.

    Returns:
        X: np.ndarray of shape (n_samples, input_window, n_features)
        y: np.ndarray of shape (n_samples, output_horizon)
        target_dates: list of timestamps corresponding to first forecast day
    """
    values = df.values
    feature_cols = [c for c in df.columns if c != target_col]
    feature_idx = [df.columns.get_loc(c) for c in feature_cols]
    target_idx = df.columns.get_loc(target_col)

    X_list, y_list, date_list = [], [], []

    total_window = model_cfg.input_window + model_cfg.output_horizon
    for i in range(len(df) - total_window):
        window = values[i : i + model_cfg.input_window]
        future = values[
            i + model_cfg.input_window : i + total_window, target_idx
        ]
        X_list.append(window[:, feature_idx])
        y_list.append(future)
        date_list.append(df.index[i + model_cfg.input_window])

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y, date_list


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[np.ndarray, ...]:
    """
    Split sequences into train/validation/test sets preserving temporal order."""
    n = len(X)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)

    X_train = X[: n - val_size - test_size]
    y_train = y[: n - val_size - test_size]

    X_val = X[n - val_size - test_size : n - test_size]
    y_val = y[n - val_size - test_size : n - test_size]

    X_test = X[n - test_size :]
    y_test = y[n - test_size :]

    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training features and apply to all splits.

    Returns scaled arrays and the fitted scaler.
    """
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    scaler = StandardScaler()
    # Fit on flattened training data
    X_train_flat = X_train.reshape(-1, n_features)
    scaler.fit(X_train_flat)

    def transform(x: np.ndarray) -> np.ndarray:
        flat = x.reshape(-1, n_features)
        scaled = scaler.transform(flat)
        return scaled.reshape(-1, n_timesteps, n_features)

    return transform(X_train), transform(X_val), transform(X_test), scaler


# ----------------------------------------------------------------------
# Model building & training
# ----------------------------------------------------------------------
def build_lstm_model(
    n_features: int,
    model_cfg: ModelConfig,
    units: int = 64,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Build and compile an LSTM-based multi-step forecasting model.

    Architecture:
        Input -> LSTM(units) -> Dense(128, relu) -> Dense(horizon)

    Args:
        n_features: Number of input features.
        model_cfg: ModelConfig with input_window and output_horizon.
        units: Number of LSTM units.
        learning_rate: Initial learning rate for Adam.

    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=(model_cfg.input_window, n_features))
    x = layers.LSTM(units, return_sequences=False)(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(model_cfg.output_horizon, name="forecast")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="lstm_forecaster")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_with_callbacks(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    model_cfg: ModelConfig,
    run_name: str,
) -> tf.keras.callbacks.History:
    """
    Train a model with early stopping and learning rate scheduling.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        batch_size: Batch size.
        model_cfg: ModelConfig.
        run_name: Unique name for logging/saving (not used to save files now).

    Returns:
        Keras History object.
    """
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=model_cfg.patience,
        restore_best_weights=True,
        verbose=1,
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=model_cfg.max_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1,
    )
    return history


def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: ModelConfig,
    search_space: SearchSpace,
    n_features: int,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Perform a simple manual grid search over hyperparameters.

    Returns:
        best_model: Keras model with lowest validation loss.
        best_params: dict containing best hyperparameters and val_loss.
    """
    best_val_loss = np.inf
    best_model = None
    best_params: Dict = {}

    run_id = 0
    for units in search_space.units_list:
        for lr in search_space.lr_list:
            for batch_size in search_space.batch_sizes:
                run_id += 1
                run_name = f"run_{run_id}_units{units}_lr{lr}_bs{batch_size}"
                print(f"\n=== Hyperparameter run: {run_name} ===")

                model = build_lstm_model(
                    n_features=n_features,
                    model_cfg=model_cfg,
                    units=units,
                    learning_rate=lr,
                )

                history = train_with_callbacks(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=batch_size,
                    model_cfg=model_cfg,
                    run_name=run_name,
                )

                val_loss = min(history.history["val_loss"])
                print(f"Run {run_name} best val_loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = {
                        "units": units,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "val_loss": float(val_loss),
                    }

    print("\n=== Best hyperparameters ===")
    print(best_params)
    return best_model, best_params


# ----------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error over all horizon steps."""
    return float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error over all horizon steps."""
    return float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    return float(
        np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + eps))) * 100.0
    )


def asymmetric_business_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    under_weight: float = 2.0,
    over_weight: float = 1.0,
) -> float:
    """
    Custom business loss metric.

    Under-forecasting (prediction < actual) is penalized more heavily than
    over-forecasting, reflecting scenarios where stockouts or capacity
    shortages are more costly than over-allocation.

    Returns:
        Weighted mean absolute error.
    """
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    diff = y_pred_flat - y_true_flat
    under_mask = diff < 0  # under-prediction
    over_mask = ~under_mask

    loss = np.zeros_like(diff)
    loss[under_mask] = under_weight * np.abs(diff[under_mask])
    loss[over_mask] = over_weight * np.abs(diff[over_mask])

    return float(np.mean(loss))


# ----------------------------------------------------------------------
# Explainability with SHAP
# ----------------------------------------------------------------------
def compute_shap_values(
    model: tf.keras.Model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    max_samples: int = 50,
) -> np.ndarray:
    """
    Compute SHAP values for an LSTM model using DeepExplainer.

    Args:
        model: Trained Keras model.
        X_background: Background dataset for SHAP, e.g., random subset of training data.
        X_explain: Samples for which explanations are required.
        max_samples: Max number of samples used for explanation.

    Returns:
        shap_values: np.ndarray of SHAP values with same shape as model output
                     plus feature dimension; for multi-output we'll focus on
                     the first horizon step.
    """
    # Down-sample for performance
    if len(X_background) > max_samples:
        X_background = X_background[:max_samples]
    if len(X_explain) > max_samples:
        X_explain = X_explain[:max_samples]

    print("Initializing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)
    # shap_values is a list when model has multiple outputs;
    # for Dense(horizon) output we get one array of shape
    # (n_samples, input_window, n_features) per horizon step.
    return shap_values


def summarize_shap_per_feature(
    shap_values: List[np.ndarray],
    feature_names: List[str],
    horizon_step: int = 0,
) -> pd.DataFrame:
    """
    Aggregate SHAP values into per-feature importance for a given horizon step.

    Args:
        shap_values: List of SHAP arrays, one per model output.
        feature_names: Names of input features.
        horizon_step: Index of forecast horizon to analyze.

    Returns:
        DataFrame with mean absolute SHAP value per feature.
    """
    sv = shap_values[horizon_step]  # (n_samples, input_window, n_features)
    # Aggregate over time and samples
    abs_sv = np.abs(sv)
    mean_importance = abs_sv.mean(axis=(0, 1))

    return pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_importance}
    ).sort_values("mean_abs_shap", ascending=False)


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main() -> None:
    """Run the full forecasting and explainability pipeline."""
    set_seed(42)

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    search_space = SearchSpace()

    print("Generating synthetic multivariate time series...")
    df = generate_synthetic_multivariate_ts(data_cfg)
    print(df.head())
    print(f"Dataset shape: {df.shape}")

    # Prepare sequences
    print("\nCreating input/output sequences...")
    X, y, target_dates = create_sequences(df, model_cfg, target_col=data_cfg.target_name)
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Train/val/test split
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
    print(
        f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
        f"Test: {X_test.shape[0]}"
    )

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    n_features = X_train_scaled.shape[2]

    # Hyperparameter search
    best_model, best_params = hyperparameter_search(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        model_cfg,
        search_space,
        n_features,
    )

    # Final evaluation on test set
    print("\nEvaluating best model on test set...")
    y_pred_test = best_model.predict(X_test_scaled)

    metrics = {
        "RMSE": rmse(y_test, y_pred_test),
        "MAE": mae(y_test, y_pred_test),
        "MAPE": mape(y_test, y_pred_test),
        "AsymmetricBusinessLoss": asymmetric_business_loss(y_test, y_pred_test),
    }

    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # SHAP explainability
    print("\nComputing SHAP values for model explainability...")
    background = X_train_scaled[:200]
    explain_samples = X_test_scaled[:60]

    shap_values = compute_shap_values(best_model, background, explain_samples)

    feature_cols = [c for c in df.columns if c != data_cfg.target_name]
    shap_summary_df = summarize_shap_per_feature(
        shap_values, feature_names=feature_cols, horizon_step=0
    )
    print("\n=== SHAP feature importance (horizon step 0) ===")
    print(shap_summary_df)

    # Optionally, save artifacts
    os.makedirs("artifacts", exist_ok=True)
    shap_summary_df.to_csv("artifacts/shap_feature_importance.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("artifacts/test_metrics.csv", index=False)

    print("\nPipeline completed. Metrics and SHAP summary saved in artifacts/.")


if __name__ == "__main__":
    main()
