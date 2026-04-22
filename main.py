import pandas as pd

import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

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
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    pred = lr_model.predict(X_test)
    proba = lr_model.predict_proba(X_test)[:, 1]

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


    #Create Neural Network
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    class SimpleMLP(torch.nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2)
            )

        def forward(self, x):
            return self.net(x)

    input_size = X_train.shape[1]

    nn_model = SimpleMLP(input_size)
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    weights = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    epochs = 50
    batch_size = 32

    for epoch in range(epochs):
        nn_model.train()

        permutation = torch.randperm(X_train_tensor.size(0))

        total_loss = 0

        for i in range(0, X_train_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train_tensor[indices]
            batch_y = y_train_tensor[indices]

            outputs = nn_model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    nn_model.eval()

    with torch.no_grad():
        outputs = nn_model(X_test_tensor)
        _, pred_nn = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)

        proba_nn = probs[:, 1]

    pred_nn = pred_nn.numpy()
    proba_nn = proba_nn.numpy()

    model_metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba)
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, pred_rf),
            "Precision": precision_score(y_test, pred_rf),
            "F1": f1_score(y_test, pred_rf),
            "ROC-AUC": roc_auc_score(y_test, proba_rf)
        },
        "XGBoost": {
            "Accuracy": accuracy_score(y_test, pred_xgb),
            "Precision": precision_score(y_test, pred_xgb),
            "F1": f1_score(y_test, pred_xgb),
            "ROC-AUC": roc_auc_score(y_test, proba_xgb)
        },
        "Neural Network": {
            "Accuracy": accuracy_score(y_test, pred_nn),
            "Precision": precision_score(y_test, pred_nn),
            "F1": f1_score(y_test, pred_nn),
            "ROC-AUC": roc_auc_score(y_test, proba_nn)
        }
    }

    feature_order = list(df3.drop("MentalHealthStatus", axis=1).columns)
    return lr_model, rf, xgb, nn_model, scaler, model_metrics, feature_order
model, rf, xgb, nn, scaler, model_metrics, feature_order = train_models()




st.set_page_config(page_title="Mental Health Predictor", layout="centered")

st.title("🧠 Mental Health Risk Predictor")
st.markdown("### Predict whether a student is at risk based on stress and mental health factors")

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


input_dict = {
    "AcademicStress": stress_scaled,
    "GAD7": gad7_scaled,
    "GPA": gpa_scaled,
    "PHQ9": phq9_scaled,
    "SleepHours": sleep_scaled
}

user_input = np.array([[input_dict[col] for col in feature_order]])
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
    if model_choice == "Neural Network":
        nn.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32)
            outputs = nn(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            proba = probs[0][1].item()
    else:
        proba = selected_model.predict_proba(user_input_scaled)[0][1]

    if proba > 0.5:
        st.success("Result: Healthy (Not At Risk)")
        st.balloons()
    else:
        st.warning("Result: At Risk")

    st.write(f"{proba:.2f}")

st.subheader(f"📊 {model_choice} Performance")

metrics = model_metrics[model_choice]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
col2.metric("Precision", f"{metrics['Precision']:.2f}")
col3.metric("F1 Score", f"{metrics['F1']:.2f}")
col4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

feature_names = ["AcademicStress", "Anxiety Score (GAD7)", "GPA", "Depression Score (PHQ9)", "SleepHours"]

st.subheader("📌 Feature Importance")

if model_choice == "Random Forest":
    importances = rf.feature_importances_
elif model_choice == "XGBoost":
    importances = xgb.feature_importances_
else:
    importances = None

if importances is not None:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))
else:
    st.info("Feature importance only available for tree-based models.")
