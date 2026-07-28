import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------
# Generate a realistic dataset
# ----------------------------
np.random.seed(42)
n = 60

hours = np.round(np.random.uniform(1, 10, n), 1)
attendance = np.round(np.random.uniform(50, 100, n), 1)
noise = np.random.normal(0, 4, n)  # real-world randomness

scores = 5 + 6 * hours + 0.3 * attendance + noise
scores = np.clip(scores, 0, 100).round(1)

df = pd.DataFrame({
    "hours": hours,
    "attendance": attendance,
    "scores": scores
})
df.to_csv("student_data.csv", index=False)
print("Dataset saved as student_data.csv")

# ----------------------------
# Prepare data
# ----------------------------
X = df[["hours", "attendance"]]
y = df["scores"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Coefficients -> hours: {model.coef_[0]:.2f}, attendance: {model.coef_[1]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# ----------------------------
# Plot: Actual vs Predicted
# ----------------------------
plt.figure()
plt.scatter(y_test, predictions, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.title(f"Actual vs Predicted (R² = {r2:.2f})")
plt.savefig("regression_line.png")
print("Saved regression_line.png")

# ----------------------------
# Plot: Hours vs Score (visual reference)
# ----------------------------
plt.figure()
plt.scatter(df["hours"], df["scores"], c=df["attendance"], cmap="viridis")
plt.colorbar(label="Attendance %")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study Hours vs Score (colored by Attendance)")
plt.savefig("graph.png")
print("Saved graph.png")

# ----------------------------
# Manual test prediction
# ----------------------------
user_hours = float(input("Enter hours studied: "))
user_attendance = float(input("Enter attendance %: "))
input_df = pd.DataFrame({"hours": [user_hours], "attendance": [user_attendance]})
predicted_score = model.predict(input_df)
print(f"Predicted Score: {predicted_score[0]:.2f}")
