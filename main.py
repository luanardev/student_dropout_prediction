import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.header("Student Dropout Prediction")
st.subheader("Please select your features!")

# get user input
input_student_type = st.selectbox('What is your student type ? 1=Matured, 2=Generic', [1, 2])
input_employment = st.selectbox('Are you working or doing business ? 0=No, 1=Yes', [0, 1])
input_withdrawal = st.selectbox('Did you had any withdrawal in the past ? 0=No, 1=Yes', [0, 1])

prediction = any;

if st.button('Make Prediction'):

    inputs = np.expand_dims([input_withdrawal, input_student_type, input_employment], 0)
    model = joblib.load("rfc_model.joblib")
    prediction = model.predict(inputs)
    st.write(f"Predicted Target is {np.squeeze(prediction, -1)}")
	
if prediction == 1:
	st.write("Student is likely to dropout")  	
elif prediction == 0:
	st.write("Student is not likely to dropout")
	
