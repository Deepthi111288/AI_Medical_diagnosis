import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Create a directory to store models if not exists
os.makedirs("models", exist_ok=True)

# List of diseases and their corresponding dataset filenames
disease_files = {
    "blood cancer": "blood_cancer.csv",
    "diabetes": "diabetes.csv",
    "cardiovascular disease": "cardiovascular_disease.csv",
    "tuberculosis": "tuberculosis.csv",
    "liver disease": "liver_disease.csv",
    "kidney disease": "kidney_disease.csv",
    "hypothyroid": "hypothyroid.csv",
    "parkinson disease": "parkinson_disease.csv"
}

# Loop through each disease and train models
for disease, file in disease_files.items():
    print(f"\nTraining model for {disease}...")

    # Check if the file exists
    if not os.path.exists(file):
        print(f"Error: File {file} not found. Skipping...")
        continue

    # Load the dataset
    df = pd.read_csv(file)
    print(f"Columns in {file}: {list(df.columns)}")

    # Ensure "Diagnosis" column exists
    if "Diagnosis" not in df.columns:
        print(f"Error: 'Diagnosis' column missing in {file}. Skipping...")
        continue

    # Drop rows where Diagnosis is missing
    df = df.dropna(subset=["Diagnosis"])

    # Separate features and target
    X = df.drop(columns=["Diagnosis"])  # Features
    y = df["Diagnosis"]  # Target variable

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(exclude=["number"]).columns

    # Handle missing values
    num_imputer = SimpleImputer(strategy="mean")  # Mean for numeric data
    cat_imputer = SimpleImputer(strategy="most_frequent")  # Most frequent for categorical data

    if len(numeric_cols) > 0:
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if len(categorical_cols) > 0:
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

        # Encode categorical features
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le

        # Save label encoders for later decoding
        with open(f"models/{disease}_feature_label_encoders.pkl", "wb") as le_file:
            pickle.dump(label_encoders, le_file)

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Save label encoder for target variable
    with open(f"models/{disease}_label_encoders.pkl", "wb") as le_file:
        pickle.dump(label_encoder, le_file)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    with open(f"models/{disease}_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model for {disease} trained and saved successfully!")

print("\nTraining process completed!")

