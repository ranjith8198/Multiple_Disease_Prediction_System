import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
# Ensure your path is correct (inside the 'data' folder)
df = pd.read_csv('data/parkinsons.csv')

# 2. Separate Features (X) and Target (Y)
# We drop 'name' because it's a text label, not a medical feature
X = df.drop(columns=['name', 'status'], axis=1)
Y = df['status']

# 3. Split the data into Training and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 4. Feature Scaling (Very important for voice frequency data)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train the Model (Using XGBoost for high performance)
model = XGBClassifier()
model.fit(X_train, Y_train)

# 6. Check Accuracy
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print(f"--- Parkinson's Model Trained ---")
print(f"Accuracy Score: {test_data_accuracy:.2%}")

# 7. SAVE THE FILES (The "Brain" and the "Scale")
joblib.dump(model, 'parkinsons_model.sav')
joblib.dump(scaler, 'parkinsons_scaler.sav')

print("Files saved: parkinsons_model.sav and parkinsons_scaler.sav")