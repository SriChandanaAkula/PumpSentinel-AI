
import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def read_sensor_file(path):
    """Read whitespace-separated hydraulic sensor readings."""
    data = pd.read_csv(path, sep=r"\s+", header=None)
    return data.apply(pd.to_numeric, errors="coerce")


def load_and_prepare(data_dir):
    # Load actual sensor readings
    vibration = read_sensor_file(os.path.join(data_dir, "VS1.txt"))
    power = read_sensor_file(os.path.join(data_dir, "EPS1.txt"))

    temperatures = [
        read_sensor_file(os.path.join(data_dir, f"TS{i}.txt"))
        for i in range(1, 5)
    ]

    # Each row represents one operating cycle
    vibration_mean = vibration.mean(axis=1)
    power_mean = power.mean(axis=1)

    temperature_mean = pd.concat(
        [sensor.mean(axis=1) for sensor in temperatures],
        axis=1
    ).mean(axis=1)

    # Actual operating-condition labels
    profile = read_sensor_file(
        os.path.join(data_dir, "profile.txt")
    )

    if profile.shape[1] != 5:
        raise ValueError("Unexpected profile.txt format.")

    # Third column: internal pump leakage
    labels = profile.iloc[:, 2].astype(int)

    # Build features
    X = pd.DataFrame({
        "vibration": vibration_mean,
        "temperature": temperature_mean,
        "power": power_mean
    })

    # Verify dataset dimensions
    if not (
        len(X) == len(labels)
        == len(vibration)
        == len(power)
    ):
        raise ValueError("Sensor files have mismatched row counts.")

    X = X.replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1) & labels.isin([0, 1, 2])

    X = X.loc[valid].reset_index(drop=True)
    y = labels.loc[valid].reset_index(drop=True)

    return X, y


def train(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    X, y = load_and_prepare(data_dir)

    label_map = {
        "HEALTHY": 0,
        "WARNING": 1,
        "CRITICAL": 2
    }

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining completed!")
    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["HEALTHY", "WARNING", "CRITICAL"],
        zero_division=0
    ))

    joblib.dump(model, os.path.join(out_dir, "pump_model.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    metadata = {
        "features": ["vibration", "temperature", "power"],
        "label_map": label_map,
        "units": {
            "vibration": "Dataset VS1 sensor units",
            "temperature": "Dataset TS1-TS4 sensor units",
            "power": "Dataset EPS1 sensor units"
        },
        "dataset": "UCI Condition Monitoring of Hydraulic Systems",
        "accuracy": float(accuracy)
    }

    with open(
        os.path.join(out_dir, "feature_meta.json"),
        "w"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nModel files saved successfully in:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="condition+monitoring+of+hydraulic+systems"
    )

    parser.add_argument("--out", default="model")

    args = parser.parse_args()

    train(args.data, args.out)
