import pandas as pd


from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import streamlit as st
import numpy as np


@st.cache_resource
def train_models():
    df = pd.read_csv('data.csv')
    df2 = pd.read_csv('mental_health_dataset.csv')

    # --- ALL YOUR PREPROCESSING HERE ---
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

    # scaling
    df["AcademicStress"] = df["AcademicStress"] / 10.0
    df2["AcademicStress"] = (df2["AcademicStress"] - 1) / 4.0

    df["SleepHours"] = (df["SleepHours"] - 3) / 7.0
    df2["SleepHours"] = (df2["SleepHours"] - 3) / 6.0

    df["GAD7"] /= 21.0
    df2["GAD7"] /= 21.0

    df["PHQ9"] /= 27.0
    df2["PHQ9"] /= 27.0

    common_cols = sorted(list(set(df.columns) & set(df2.columns)))

    df3 = pd.concat([df[common_cols], df2[common_cols]]).reset_index(drop=True)

    X = df3.drop("MentalHealthStatus", axis=1).values
    y = df3["MentalHealthStatus"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    proba_rf = rf.predict_proba(X_test)[:, 1]

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1]))
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    proba_xgb = xgb.predict_proba(X_test)[:, 1]

    nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
    nn.fit(X_train, y_train)
    pred_nn = nn.predict(X_test)
    proba_nn = nn.predict_proba(X_test)[:, 1]

    model_metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba)
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, pred_rf),
            "F1": f1_score(y_test, pred_rf),
            "ROC-AUC": roc_auc_score(y_test, proba_rf)
        },
        "XGBoost": {
            "Accuracy": accuracy_score(y_test, pred_xgb),
            "F1": f1_score(y_test, pred_xgb),
            "ROC-AUC": roc_auc_score(y_test, proba_xgb)
        },
        "Neural Network": {
            "Accuracy": accuracy_score(y_test, pred_nn),
            "F1": f1_score(y_test, pred_nn),
            "ROC-AUC": roc_auc_score(y_test, proba_nn)
        }
    }

    return  model, rf, xgb, nn, scaler, model_metrics
model, rf, xgb, nn, scaler, model_metrics = train_models()



st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health Risk Predictor")
st.markdown("### Predict whether a student is at risk based on lifestyle and mental health factors")

model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
)


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


if model_choice == "Logistic Regression":
    selected_model = model
elif model_choice == "Random Forest":
    selected_model = rf
elif model_choice == "XGBoost":
    selected_model = xgb
else:
    selected_model = nn


if st.button("Predict"):
    proba = selected_model.predict_proba(user_input_scaled)[0][1]
    if proba > 0.5:
        st.success("Result: Healthy (Not At Risk)")
        st.balloons()
        st.write(f"{proba:.2f}")
    else:
        st.warning("Result: At Risk")
        st.write(f"{proba:.2f}")

st.subheader(f"📊 {model_choice} Performance")

metrics = model_metrics[model_choice]

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
col2.metric("F1 Score", f"{metrics['F1']:.2f}")
col3.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")
