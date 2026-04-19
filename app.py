import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# Title
# ----------------------------
st.title("🎓 Student Score Predictor")

st.write("Enter study hours to predict score")

# ----------------------------
# Dataset
# ----------------------------
data = {
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "scores": [10,20,30,40,50,60,65,70,80,95]
}

df = pd.DataFrame(data)

# ----------------------------
# Train Model
# ----------------------------
X = df[["hours"]]
y = df["scores"]

model = LinearRegression()
model.fit(X, y)

# ----------------------------
# User Input
# ----------------------------
hours = st.number_input("Enter hours studied", min_value=0.0, max_value=12.0, step=0.5)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame({"hours": [hours]})
    prediction = model.predict(input_df)

    st.success(f"Predicted Score: {prediction[0]:.2f}")

# ----------------------------
# Show Graph
# ----------------------------
fig, ax = plt.subplots()
ax.scatter(df["hours"], df["scores"])
ax.plot(df["hours"], model.predict(X), linestyle="--")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Score")
ax.set_title("Study Hours vs Score")

st.pyplot(fig)