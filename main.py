import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.header("Student Dropout Prediction")
st.subheader("Please select your features!")

#options
STUDENTTYPE = {1: "GENERIC", 2: "MATURED"}
EMPLOYMENT = {0:"SELF EMPLOYED", 1: "NOT EMPLOYED", 2: "EMPLOYED"}
WITHDRAWAL = {0: "YES", 1: "NO"}
REPEAT = {0: "YES", 1: "NO"}
MARITALSTATUS = {1: "Single", 2: "Married"}
GENDER = {1: "Male", 2: "Female"}

# get user input
input_student_type = st.selectbox('What is your student type ?', STUDENTTYPE.keys(), format_func=lambda x:STUDENTTYPE[ x ])
input_gender = st.selectbox('What is your gender ?', GENDER.keys(), format_func=lambda x:GENDER[ x ])
input_age = st.text_input('What is your age ?')
input_marital = st.selectbox('What is your marital status ?', MARITALSTATUS.keys(), format_func=lambda x:MARITALSTATUS[ x ])
input_employment = st.selectbox('What is your employment status ?', EMPLOYMENT.keys(), format_func=lambda x:EMPLOYMENT[ x ])
input_withdrawal = st.selectbox('Did you withdrawal in the past ?', WITHDRAWAL.keys(), format_func=lambda x:WITHDRAWAL[ x ])
input_repeat = st.selectbox('Did you repeat in the past ?', REPEAT.keys(), format_func=lambda x:REPEAT[ x ])

prediction = any

if st.button('Make Prediction'):

    inputs = np.expand_dims([
	    input_withdrawal, 
	    input_repeat, 
	    input_marital, 
	    input_gender,
	    input_student_type, 
	    input_age,
	    input_employment], 0)
    
    model = joblib.load("rfc_model.joblib")
    prediction = model.predict(inputs)
    st.write(f"Predicted Target is {np.squeeze(prediction, -1)}")
	
if prediction == 1:
	st.write("Student is likely to dropout")  	
elif prediction == 0:
	st.write("Student is not likely to dropout")
	
