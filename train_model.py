import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("user_behavior_dataset.csv")

# Target column
target = "User Behavior Class"

X = data.drop(columns=[target])
y = data[target]

# Categorical and numerical columns
categorical_features = [
    "Device Model",
    "Operating System",
    "Gender"
]

numerical_features = [
    "User ID",
    "App Usage Time (min/day)",
    "Screen On Time (hours/day)",
    "Battery Drain (mAh/day)",
    "Number of Apps Installed",
    "Data Usage (MB/day)",
    "Age"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("User Behavior ML Model")
print("----------------------")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")

# Save model
joblib.dump(
    pipeline,
    "user_behavior_model.pkl"
)

# Save metrics
metrics = {
    "accuracy": round(accuracy, 4),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

# Save dataset used for the run
data.to_csv(
    "user_behavior_results.csv",
    index=False
)

print("\nGenerated files:")
print("user_behavior_model.pkl")
print("metrics.json")
print("user_behavior_results.csv")
