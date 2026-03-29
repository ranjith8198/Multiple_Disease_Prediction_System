Project Overview

This system is a Multiple Disease Prediction System designed to assist healthcare professionals in early diagnosis. It utilizes three distinct Machine Learning models to analyze patient data and provide instant predictions.

Key Results

Parkinson's Disease: Achieved 84.62% accuracy using vocal frequency features.

Liver Disease: Achieved 70.09% accuracy analyzing enzymes and bilirubin levels.

Kidney Disease: Achieved 100.00% accuracy identifying Chronic Kidney Disease (CKD) indicators.

Technical Implementation

Data Preprocessing: Handled missing values (Median Imputation) and categorical encoding (Label Encoding).

Feature Scaling: Used StandardScaler to normalize medical units across different tests.

Modeling: * XGBoost: Used for Parkinson's and Kidney data for high-performance classification.

Random Forest: Used for Liver data to handle overlapping feature sets.

UI/UX: Developed an interactive multi-page web app using Streamlit.