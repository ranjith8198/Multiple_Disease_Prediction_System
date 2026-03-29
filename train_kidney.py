import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('data/kidney_disease.csv')

# 2. Cleaning (Kidney data has lots of text and missing values)
df = df.drop('id', axis=1)
df['classification'] = df['classification'].replace(to_replace={'ckd\t':'ckd', 'notckd': 'not ckd'})
df['classification'] = df['classification'].map({'ckd': 1, 'not ckd': 0})

# Fill missing values and encode text
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.Categorical(df[col]).codes
df = df.fillna(df.median())

# 3. Features and Target
X = df.drop(columns=['classification'], axis=1)
Y = df['classification']

# 4. Split and Scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train
model = XGBClassifier()
model.fit(X_train, Y_train)

# 6. Save
print(f"Kidney Model Accuracy: {accuracy_score(Y_test, model.predict(X_test)):.2%}")
joblib.dump(model, 'kidney_model.sav')
joblib.dump(scaler, 'kidney_scaler.sav')
print("Kidney files saved successfully!")