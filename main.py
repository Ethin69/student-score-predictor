import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------
# Create dataset
# ----------------------------
data = {
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "scores": [10,20,30,40,50,60,65,70,80,95]
}

df = pd.DataFrame(data)

# ----------------------------
# Visualization (save graph)
# ----------------------------
plt.scatter(df["hours"], df["scores"])
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study Hours vs Score")
plt.savefig("graph.png")
print("Graph saved as graph.png")

# ----------------------------
# Prepare data
# ----------------------------
X = df[["hours"]]
y = df["scores"]

# ----------------------------
# Train-test split (fixed randomness)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Predict & evaluate
# ----------------------------
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Error:", mse)

# ----------------------------
# Plot regression line (pro touch)
# ----------------------------
plt.figure()
plt.scatter(df["hours"], df["scores"])
plt.plot(df["hours"], model.predict(X), linestyle="--")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Study Hours vs Score (Model Fit)")
plt.savefig("regression_line.png")
print("Regression plot saved as regression_line.png")

# ----------------------------
# User input (no warning)
# ----------------------------
user_hours = float(input("Enter hours studied: "))
input_df = pd.DataFrame({"hours": [user_hours]})

predicted_score = model.predict(input_df)
print("Predicted Score:", predicted_score[0])