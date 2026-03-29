import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('data/indian_liver_patient.csv')

# 2. Handle Missing Values
# Liver data often has nulls in 'Albumin_and_Globulin_Ratio'
df = df.fillna(method='ffill') 

# 3. Encode Categorical Data (Male/Female -> 1/0)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 4. Features and Target
# Usually, the target column is 'Dataset' (1=Sick, 2=Healthy)
X = df.drop(columns=['Dataset'], axis=1)
Y = df['Dataset']

# 5. Split and Scale
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train (Random Forest works very well for Liver data)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# 7. Evaluate and Save
Y_pred = model.predict(X_test)
print(f"Liver Model Accuracy: {accuracy_score(Y_test, Y_pred):.2%}")

joblib.dump(model, 'liver_model.sav')
joblib.dump(scaler, 'liver_scaler.sav')
print("Liver files saved successfully!")