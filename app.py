import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# --- LOAD THE MODELS ---
# Using joblib as we did in our training scripts
parkinsons_model = joblib.load('parkinsons_model.sav')
parkinsons_scaler = joblib.load('parkinsons_scaler.sav')

liver_model = joblib.load('liver_model.sav')
liver_scaler = joblib.load('liver_scaler.sav')

kidney_model = joblib.load('kidney_model.sav')
kidney_scaler = joblib.load('kidney_scaler.sav')

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Parkinsons Prediction',
                           'Liver Disease Prediction',
                           'Kidney Disease Prediction'],
                          icons=['person', 'heart', 'activity'],
                          default_index=0)

# --- PARKINSONS SECTION ---
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1: fo = st.text_input('MDVP:Fo(Hz)')
    with col2: fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3: flo = st.text_input('MDVP:Flo(Hz)')
    with col1: jitter = st.text_input('MDVP:Jitter(%)')
    with col2: rap = st.text_input('MDVP:RAP')
    with col3: ppq = st.text_input('MDVP:PPQ')
    with col1: ddp = st.text_input('Jitter:DDP')
    with col2: shimmer = st.text_input('MDVP:Shimmer')
    
    parkinsons_diag = ''
    if st.button("Parkinson's Test Result"):
        user_input = [fo, fhi, flo, jitter, rap, ppq, ddp, shimmer, 0.05, 0.02, 0.03, 0.04, 0.05, 0.01, 21.0, 0.4, 0.5, 0.6, 0.2, 2.3, 0.1, 0.2] # Placeholders for full 22 features
        user_input = [float(x) for x in user_input]
        std_data = parkinsons_scaler.transform([user_input])
        prediction = parkinsons_model.predict(std_data)
        
        if prediction[0] == 1:
            st.error("The person has Parkinson's disease")
        else:
            st.success("The person does not have Parkinson's disease")

# --- LIVER SECTION ---
if selected == 'Liver Disease Prediction':
    st.title("Liver Disease Prediction using ML")
    col1, col2 = st.columns(2)
    
    with col1: age = st.text_input('Age')
    with col2: gender = st.selectbox('Gender', ('Male', 'Female'))
    with col1: tb = st.text_input('Total Bilirubin')
    with col2: db = st.text_input('Direct Bilirubin')
    with col1: alkphos = st.text_input('Alkaline Phosphotase')
    with col2: sgpt = st.text_input('Sgpt Alamine Aminotransferase')
    with col1: sgot = st.text_input('Sgot Aspartate Aminotransferase')
    with col2: tp = st.text_input('Total Proteins')
    with col1: alb = st.text_input('Albumin')
    with col2: agr = st.text_input('Albumin and Globulin Ratio')

    if st.button("Liver Test Result"):
        gender_encoded = 1 if gender == 'Male' else 0
        user_input = [age, gender_encoded, tb, db, alkphos, sgpt, sgot, tp, alb, agr]
        user_input = [float(x) for x in user_input]
        std_data = liver_scaler.transform([user_input])
        prediction = liver_model.predict(std_data)
        
        if prediction[0] == 1:
            st.error("The person has Liver disease")
        else:
            st.success("The person does not have Liver disease")

# --- KIDNEY SECTION ---
# --- KIDNEY SECTION ---
if selected == 'Kidney Disease Prediction':
    st.title("Kidney Disease Prediction using ML")
    
    col1, col2, col3 = st.columns(3)
    
    with col1: age = st.text_input('Age')
    with col2: bp = st.text_input('Blood Pressure')
    with col3: al = st.text_input('Albumin')
    with col1: su = st.text_input('Sugar')
    with col2: rbc = st.selectbox('Red Blood Cells', ('Normal', 'Abnormal'))
    with col3: pc = st.selectbox('Pus Cell', ('Normal', 'Abnormal'))
    with col1: pcc = st.selectbox('Pus Cell Clumps', ('Present', 'Not Present'))
    with col2: ba = st.selectbox('Bacteria', ('Present', 'Not Present'))
    with col3: bgr = st.text_input('Blood Glucose Random')
    with col1: bu = st.text_input('Blood Urea')
    with col2: sc = st.text_input('Serum Creatinine')

    if st.button("Kidney Test Result"):
        # Encoding categorical choices to match the trainer
        rbc_num = 1 if rbc == 'Normal' else 0
        pc_num = 1 if pc == 'Normal' else 0
        pcc_num = 1 if pcc == 'Present' else 0
        ba_num = 1 if ba == 'Present' else 0
        
        # We use placeholders for the remaining hidden features to match the 24 required by the model
        user_input = [age, bp, al, su, rbc_num, pc_num, pcc_num, ba_num, bgr, bu, sc, 137, 4.6, 15, 44, 7800, 5.2, 0, 0, 0, 1, 0, 0, 0]
        user_input = [float(x) for x in user_input]
        
        std_data = kidney_scaler.transform([user_input])
        prediction = kidney_model.predict(std_data)
        
        if prediction[0] == 1:
            st.error("The person has Chronic Kidney Disease")
        else:
            st.success("The person does not have Chronic Kidney Disease")
        