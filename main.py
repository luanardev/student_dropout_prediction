import joblib
import streamlit as st
import pandas as pd
import numpy as np

st.header("Student Dropout Prediction")

st.text_input("Enter your Name: ", key="name")

# load model
model = joblib.load("rfc_model.joblib")

st.subheader("Please select your relevant features!")

input_st = st.selectBox('What is your student type ?', ("Generic", "Matured"))
input_es = st.selectBox('Are you working or doing business ?', ("Yes", "No"))
input_wh = st.selectBox('Did you had any withdrawal in the past ?', ("Yes", "No"))

if st.button('Make Prediction'):
    inputs = np.expand_dims([input_st, input_wh, input_es], 0)
    prediction = model.predict(inputs)
    print("Predicted Target = ", np.squeeze(prediction, -1))

if prediction == 1:
    print("Student is likely to dropout")

elif prediction == 0:
    print("Student is not likely to dropout")

st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
