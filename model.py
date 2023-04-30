# import libraries
import os
import joblib
import pandas as pd
import load_data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


# LOAD DATA FRAME
processed_csv = load_data.process('students_dataset.csv')
df = pd.read_csv(processed_csv)

# SPLIT DATA FRAME FOR MODEL FITTING
array = df.values
X = array[:, 0:3]
y = array[:, 3]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15, random_state=1)

# prepare RF model
model = RandomForestClassifier()

# Make predictions on validation dataset
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
confusion_matrix(Y_validation, predictions)

# OverSampling with SMOTE
sm = SMOTE(random_state=42)
X_train_ovr, y_train_ovr = sm.fit_resample(X_train, Y_train.ravel())

# Make predictions on validation dataset
model.fit(X_train_ovr, y_train_ovr.ravel())
predictions = model.predict(X_validation)
confusion_matrix(Y_validation, predictions)

# UnderSampling with SMOTE
under_sm = RandomUnderSampler(random_state=42)
X_train_undr, y_train_undr = under_sm.fit_resample(X_train, Y_train.ravel())

# Make predictions on validation dataset
model.fit(X_train_undr, y_train_undr.ravel())
predictions = model.predict(X_validation)
confusion_matrix(Y_validation, predictions)

# HYPERPARAMETER OPTIMIZATION

# -----RF GRID SEARCH-----

# define evaluation
cv = RepeatedStratifiedKFold()

# define search space
space = {
    'n_estimators': [115, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(2, 20, 1),
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'min_samples_split': [1, 2, 5, 4, 6]
}
# define search
search = GridSearchCV(estimator=model, param_grid=space, scoring='accuracy', cv=3, n_jobs=-1, verbose=2)
# execute search
result = search.fit(X_train_undr, y_train_undr.ravel())
# summarize result
print('.....................................................................\n')
print("Grid search results for RF")
print('Best Score: %s' % result.best_score_)
print('Best Hyper parameters: %s' % result.best_params_)
print('.....................................................................\n')

# Rebuild model with optimized parameters
rfc_model = RandomForestClassifier(criterion='gini', max_depth=2, min_samples_leaf=3, min_samples_split=6, n_estimators=115)
rfc_model.fit(X_train_undr, y_train_undr.ravel())
predictions = rfc_model.predict(X_validation)
confusion_matrix(Y_validation, predictions)

# save
joblib.dump(rfc_model, "rfc_model.joblib")

print("Model training completed")
