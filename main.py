import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.header("Student Dropout Prediction")
st.subheader("Please select your features!")

#options
STUDENTTYPE = {1: "MATURED", 2: "GENERIC"}
EMPLOYMENT = {0: "NO", 1: "YES"}
WITHDRAWAL = {0: "NO", 1: "YES"}
REPEAT = {0: "NO", 1: "YES"}
MARITALSTATUS = {4: "Male", 2: "Female"}
GENDER = {2: "Male", 1: "Female"}


# get user input
input_student_type = st.selectbox('What is your student type ?', STUDENTTYPE.keys(), format_func=lambda x:STUDENTTYPE[ x ])
input_gender = st.selectbox('What is your gender ?', GENDER.keys(), format_func=lambda x:GENDER[ x ])
input_age = st.text_input('What is your age ?')
input_marital = st.selectbox('What is your marital status ?', MARITALSTATUS.keys(), format_func=lambda x:MARITALSTATUS[ x ])
input_employment = st.selectbox('Are you working or doing business ?', EMPLOYMENT.keys(), format_func=lambda x:EMPLOYMENT[ x ])
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
	
