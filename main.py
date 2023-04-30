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

# get user input
input_student_type = st.selectbox('What is your student type ?', STUDENTTYPE.keys(), format_func=lambda x:STUDENTTYPE[ x ])
input_employment = st.selectbox('Are you working or doing business ?', EMPLOYMENT.keys(), format_func=lambda x:EMPLOYMENT[ x ])
input_withdrawal = st.selectbox('Did you withdrawal in the past ?', WITHDRAWAL.keys(), format_func=lambda x:WITHDRAWAL[ x ])

prediction = any

if st.button('Make Prediction'):

    inputs = np.expand_dims([input_withdrawal, input_student_type, input_employment], 0)
    model = joblib.load("rfc_model.joblib")
    prediction = model.predict(inputs)
    st.write(f"Predicted Target is {np.squeeze(prediction, -1)}")
	
if prediction == 1:
	st.write("Student is likely to dropout")  	
elif prediction == 0:
	st.write("Student is not likely to dropout")
	
