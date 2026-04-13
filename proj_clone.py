import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import time


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st





df= pd.read_csv('data.csv') # loading dataset
df2= pd.read_csv('mental_health_dataset.csv') # loading dataset


df2["Mental_Health_Status"] = df2["Mental_Health_Status"].replace({
    0: 1,
    1: 0,
    2: 0
})

df2 = df2.drop(columns=["Student_ID", "Daily_Reflections", "Mood_Description"])

df2 = df2.rename(columns={
    "Sleep_Hours": "SleepHours",
    "Stress_Level": "AcademicStress",
    "Anxiety_Score": "GAD7",
    "Depression_Score": "PHQ9",
    "Mental_Health_Status": "MentalHealthStatus"
})

df["AcademicStress"] = df["AcademicStress"] / 10.0

df2["AcademicStress"] = (df2["AcademicStress"] - 1) / 4.0



df["SleepHours"] = (df["SleepHours"] - 3) / 7.0

df2["SleepHours"] = (df2["SleepHours"] - 3) / 6.0
df["GAD7"] = df["GAD7"] / 21.0
df2["GAD7"] = df2["GAD7"] / 21.0

df["PHQ9"] = df["PHQ9"] / 27.0
df2["PHQ9"] = df2["PHQ9"] / 27.0

common_cols = sorted(list(set(df.columns) & set(df2.columns)))

df1_aligned = df[common_cols]
df2_aligned = df2[common_cols]

df3 = pd.concat([df1_aligned, df2_aligned], axis=0).reset_index(drop=True)

print(df3)

X = df3.drop("MentalHealthStatus", axis=1).values
y = df3["MentalHealthStatus"].values





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))

print("ROC-AUC:", roc_auc_score(y_test, proba))

import streamlit as st
import numpy as np

st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health Risk Predictor")
st.markdown("### Predict whether a student is at risk based on lifestyle and mental health factors")

sleep = st.slider("Sleep hours", 3, 10)
academic_stress = st.slider("Academic Stress Level", 0, 10)
gad7 = st.slider("Anxiety Score (GAD7)", 0, 21)
phq9 = st.slider("Depression Score (PHQ9)", 0, 27)
gpa = st.number_input("GPA", min_value=0.0, max_value=5.0, step=0.01)


sleep_scaled = (sleep - 3) / 7.0
stress_scaled = academic_stress / 10.0
gad7_scaled = gad7 / 21.0
phq9_scaled = phq9 / 27.0
gpa_scaled = gpa / 5.0


user_input = np.array([[
    stress_scaled,
    gad7_scaled,
    gpa_scaled,
    phq9_scaled,
    sleep_scaled
]])
user_input_scaled = scaler.transform(user_input)
print(user_input_scaled)

if st.button("Predict"):
    proba = model.predict_proba(user_input_scaled)[0][1]
    if proba > 0.5:
        st.success("Result: Healthy (Not At Risk)")
        st.balloons()
        st.write(f"{proba:.2f}")
    else:
        st.warning("Result: At Risk")
        st.write(f"{proba:.2f}")

