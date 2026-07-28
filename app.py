import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Student Score Predictor", page_icon="🎓")

st.title("🎓 Student Score Predictor")
st.write("Predicts a student's score using **hours studied** and **attendance %**, powered by multi-variable Linear Regression.")

# ----------------------------
# Generate & cache the dataset + model
# ----------------------------
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 60
    hours = np.round(np.random.uniform(1, 10, n), 1)
    attendance = np.round(np.random.uniform(50, 100, n), 1)
    noise = np.random.normal(0, 4, n)
    scores = 5 + 6 * hours + 0.3 * attendance + noise
    scores = np.clip(scores, 0, 100).round(1)

    df = pd.DataFrame({"hours": hours, "attendance": attendance, "scores": scores})
    X = df[["hours", "attendance"]]
    y = df["scores"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    return model, df, r2, mae

model, df, r2, mae = train_model()

# ----------------------------
# Model performance
# ----------------------------
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.2f}")
col2.metric("Mean Absolute Error", f"{mae:.2f} pts")

st.divider()

# ----------------------------
# User input
# ----------------------------
st.subheader("Try a Prediction")
c1, c2 = st.columns(2)
hours = c1.number_input("Hours studied", min_value=0.0, max_value=12.0, step=0.5, value=5.0)
attendance = c2.number_input("Attendance %", min_value=0.0, max_value=100.0, step=1.0, value=75.0)

if st.button("Predict Score"):
    input_df = pd.DataFrame({"hours": [hours], "attendance": [attendance]})
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Score: {prediction:.2f}")

st.divider()

# ----------------------------
# Visualization
# ----------------------------
st.subheader("Data Overview")
fig, ax = plt.subplots()
scatter = ax.scatter(df["hours"], df["scores"], c=df["attendance"], cmap="viridis")
plt.colorbar(scatter, label="Attendance %")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Score")
ax.set_title("Study Hours vs Score (colored by Attendance)")
st.pyplot(fig)
