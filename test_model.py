
import os
import subprocess
import sys
import joblib
import pandas as pd

MODEL_PATH = "artifacts/ecommerce_pipeline.joblib"
DATA_PATH = "data/mlops_975_dataset.csv"

def test_dataset_exists():
    assert os.path.exists(DATA_PATH)

def test_dataset_has_target():
    df = pd.read_csv(DATA_PATH)
    assert "high_value_purchase" in df.columns

def test_training_script_runs():
    result = subprocess.run(
        [sys.executable, "train_model.py"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, result.stderr

def test_saved_model_predicts():
    assert os.path.exists(MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customer_id", "high_value_purchase"])
    predictions = model.predict(X.head(5))
    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})
