import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.header("Student Dropout Prediction")
st.subheader("Please select your features!")

#options
STUDENTTYPE = {1: "Generic", 2: "Matured"}
EMPLOYMENT = {0:"Self Employed", 1: "Not Employed", 2: "Employed"}
WITHDRAWAL = {0: "Yes", 1: "No"}
REPEAT = {0: "Yes", 1: "No"}
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
    #st.write(f"Predicted Target is {np.squeeze(prediction, -1)}")
	
if prediction == 1:
	st.write("STUDENT IS LIKELY TO DROPOUT")  	
elif prediction == 0:
	st.write("STUDENT IS NOT LIKELY TO DROPOUT")
	
