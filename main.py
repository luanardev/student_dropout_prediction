import joblib
import streamlit as st
import pandas as pd
import numpy as np


st.header("Student Dropout Prediction")

st.text_input("Enter your Name: ", key="name")

data = pd.read_csv("processed_data.csv")

# load model
model = joblib.load("rfc_model.joblib")

st.subheader("Please select your relevant features!")

input_st = st.slider('Student Type(ST))', 0.0, max(data["StudentType(ST)"]), 1.0)
input_wh = st.slider('Withdrawal History(WH)', 0.0, max(data["Withdrawal History(WH)"]), 1.0)
input_es = st.slider('Employment Status(ES)', 0.0, max(data["EmploymentStatus(ES)"]), 1.0)

if st.button('Make Prediction'):
    inputs = np.expand_dims([input_st, input_wh, input_es], 0)
    prediction = model.predict(inputs)
    print("Predicted Target = ", np.squeeze(prediction, -1))

	if prediction == 1:
		print("Student is likely to dropout")
	elif prediction == 0:
		print("Student is not likely to dropout")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")



