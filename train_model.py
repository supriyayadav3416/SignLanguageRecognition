
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file = "data/sign_data.csv"

if not os.path.exists(file):
    print("Dataset file not found!")
    raise SystemExit

df = pd.read_csv(file, header=None)

print("Total samples:", len(df))
print("Letters collected:", sorted(df[0].unique()))

if len(df) < 26 or df[0].nunique() < 2:
    print("Please collect data for at least two letters.")
    raise SystemExit

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

counts = y.value_counts()
if counts.min() < 2:
    print("Collect at least 2 samples for every letter.")
    raise SystemExit

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

joblib.dump(model, "sign_model.pkl")

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("Model saved successfully as sign_model.pkl")
