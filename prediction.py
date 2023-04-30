import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('processed_data.csv')

array = data.values
X = array[:, 0:3]
Y = array[:, 3]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# load model
print("====== Loading Model ======= \n")
rfc_model = joblib.load("rfc_model.joblib")
pred = rfc_model.predict(X_train)

# %%
print("===== Test Data ===== \n")
testData = [[1, 3, 0]]
print(testData)

print("===== Predicting ==== \n")
prediction = rfc_model.predict(testData)

print("final prediction", np.squeeze(prediction, -1))

if prediction == 1:
    print("Student is not likely to dropout")
elif prediction == 0:
    print("Student is likely to dropout")
