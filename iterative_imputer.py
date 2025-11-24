"""
Iterative Imputer for Cattle Data

This script uses sklearn's IterativeImputer to impute missing values
in both training and test datasets. The imputer is fitted on the training
data and then applied to both datasets to maintain consistency.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# File paths
TRAIN_PATH = "cattle_data_train.csv"
TEST_PATH = "cattle_data_test.csv"
OUTPUT_TRAIN_PATH = "cattle_data_imputed_train.csv"
OUTPUT_TEST_PATH = "cattle_data_imputed_test.csv"

print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Identify numerical columns (IterativeImputer works on numerical data)
train_numerical_cols = train.select_dtypes(include=[np.number]).columns.tolist()
test_numerical_cols = test.select_dtypes(include=[np.number]).columns.tolist()

# Use only columns that exist in both datasets (exclude target variable from test)
numerical_cols = [col for col in train_numerical_cols if col in test_numerical_cols]
print(f"\nNumerical columns (common to both datasets): {len(numerical_cols)}")
print(f"Training-only numerical columns: {set(train_numerical_cols) - set(numerical_cols)}")

# Check missing values before imputation
print("\nMissing values in training data (before imputation):")
train_missing = train[numerical_cols].isna().sum()
print(train_missing[train_missing > 0])

print("\nMissing values in test data (before imputation):")
test_missing = test[numerical_cols].isna().sum()
print(test_missing[test_missing > 0])

# Separate numerical and non-numerical columns
# For train: preserve target variable (Milk_Yield_L) separately if it exists
target_col = 'Milk_Yield_L' if 'Milk_Yield_L' in train.columns else None

train_numerical = train[numerical_cols].copy()
if target_col:
    train_target = train[[target_col]].copy()
    train_non_numerical = train.drop(columns=numerical_cols + [target_col]).copy()
else:
    train_target = None
    train_non_numerical = train.drop(columns=numerical_cols).copy()

test_numerical = test[numerical_cols].copy()
test_non_numerical = test.drop(columns=numerical_cols).copy()

# Create and fit IterativeImputer on training data
print("\nFitting IterativeImputer on training data...")
imputer = IterativeImputer(
    random_state=42,
    max_iter=10,
    imputation_order='ascending'
)

# Fit on training data
imputer.fit(train_numerical)

# Transform both train and test
print("Imputing training data...")
train_numerical_imputed = pd.DataFrame(
    imputer.transform(train_numerical),
    columns=numerical_cols,
    index=train_numerical.index
)

print("Imputing test data...")
test_numerical_imputed = pd.DataFrame(
    imputer.transform(test_numerical),
    columns=numerical_cols,
    index=test_numerical.index
)

# Combine numerical and non-numerical columns back
# Ensure column order matches original
if target_col:
    train_imputed = pd.concat([train_non_numerical, train_numerical_imputed, train_target], axis=1)
else:
    train_imputed = pd.concat([train_non_numerical, train_numerical_imputed], axis=1)
train_imputed = train_imputed[train.columns]  # Reorder to match original

test_imputed = pd.concat([test_non_numerical, test_numerical_imputed], axis=1)
test_imputed = test_imputed[test.columns]  # Reorder to match original

# Verify no missing values remain
print("\nMissing values in training data (after imputation):")
train_missing_after = train_imputed[numerical_cols].isna().sum()
print(train_missing_after[train_missing_after > 0])

print("\nMissing values in test data (after imputation):")
test_missing_after = test_imputed[numerical_cols].isna().sum()
print(test_missing_after[test_missing_after > 0])

# Save imputed datasets
print(f"\nSaving imputed training data to {OUTPUT_TRAIN_PATH}...")
train_imputed.to_csv(OUTPUT_TRAIN_PATH, index=False)

print(f"Saving imputed test data to {OUTPUT_TEST_PATH}...")
test_imputed.to_csv(OUTPUT_TEST_PATH, index=False)

print("\nImputation complete!")
print(f"Imputed training data shape: {train_imputed.shape}")
print(f"Imputed test data shape: {test_imputed.shape}")

