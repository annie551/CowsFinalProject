"""
Generate a submission file using saved models.

Supports both:
- .pkl files: sklearn Pipeline models (preprocessor + estimator)
- .pth files: PyTorch neural network models (requires separate preprocessor)

For PyTorch models, the preprocessor should be saved as preprocessor_fold_X.pkl
in the same directory as the model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models_nn" / "neuralnetwork_fold_1.pth"
TEST_PATH = PROJECT_ROOT / "cleaned_test_nn.csv"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


# PyTorch Neural Network Model (same as in neural_network.ipynb)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='relu', dropout=0.0):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (regression)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def load_pytorch_model(model_path: Path, device=None):
    """Load a PyTorch model from .pth file"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading PyTorch model from {model_path} ...")
    
    # Set device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_params = checkpoint['model_params']
    hyperparams = checkpoint.get('hyperparams', {})
    
    # Reconstruct model
    model = NeuralNetwork(
        input_size=model_params['input_size'],
        hidden_sizes=model_params['hidden_sizes'],
        activation=model_params['activation'],
        dropout=0.0
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load preprocessor (should be in same directory with same fold number)
    # Model filename format: neuralnetwork_fold_X.pth
    # Preprocessor filename format: preprocessor_fold_X.pkl
    fold_part = '_'.join(model_path.stem.split('_')[-2:])  # Extract "fold_X"
    preprocessor_path = model_path.parent / f"preprocessor_{fold_part}.pkl"
    
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {preprocessor_path}. "
            f"Expected filename: preprocessor_{fold_part}.pkl"
        )
    
    print(f"Loading preprocessor from {preprocessor_path} ...")
    preprocessor = joblib.load(preprocessor_path)
    
    return model, preprocessor, device


def load_sklearn_model(model_path: Path):
    """Load a sklearn Pipeline model from .pkl file"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading sklearn model from {model_path} ...")
    return joblib.load(model_path)


def load_test_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test data file not found at {path}")
    print(f"Loading test data from {path} ...")
    return pd.read_csv(path)


def create_submission(test_df: pd.DataFrame, predictions) -> pd.DataFrame:
    id_col = "Cattle_ID" if "Cattle_ID" in test_df.columns else None
    if id_col:
        submission = pd.DataFrame(
            {
                "Cattle_ID": test_df[id_col],
                "Milk_Yield_L": predictions,
            }
        )
    else:
        submission = pd.DataFrame(
            {
                "Cattle_ID": range(1, len(predictions) + 1),
                "Milk_Yield_L": predictions,
            }
        )
    return submission


def predict_pytorch(model, preprocessor, feature_df, device):
    """Make predictions using PyTorch model"""
    # Preprocess features
    X_processed = preprocessor.transform(feature_df)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_processed).to(device)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()
    
    return predictions


def predict_sklearn(model, feature_df):
    """Make predictions using sklearn model"""
    return model.predict(feature_df)


def main():
    # Determine model type from file extension
    model_path = MODEL_PATH
    is_pytorch = model_path.suffix == '.pth'
    
    # Load model
    if is_pytorch:
        model, preprocessor, device = load_pytorch_model(model_path)
    else:
        model = load_sklearn_model(model_path)
        preprocessor = None
        device = None
    
    # Load test data
    test_df = load_test_data(TEST_PATH)
    
    # Prepare features
    id_cols = ["Cattle_ID"] if "Cattle_ID" in test_df.columns else []
    feature_df = test_df.drop(columns=id_cols, errors="ignore")
    
    # Generate predictions
    print("Generating predictions ...")
    if is_pytorch:
        predictions = predict_pytorch(model, preprocessor, feature_df, device)
    else:
        predictions = predict_sklearn(model, feature_df)
    
    # Create submission
    submission = create_submission(test_df, predictions)
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()

