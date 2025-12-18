import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student GPA Predictor", layout="centered")

st.title("🎓 Student GPA Predictor")

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("student_details.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# 2. Prepare data
# --------------------------------------------------
X = df.drop("GPA", axis=1)
y = df["GPA"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

# --------------------------------------------------
# 3. Train model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train_ss, y_train)

# --------------------------------------------------
# 4. Model evaluation
# --------------------------------------------------
y_pred = model.predict(X_test_ss)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"**Mean Squared Error:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# --------------------------------------------------
# 5. Sidebar: User input
# --------------------------------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.number_input("Age", min_value=15, max_value=60, value=20)
credits = st.sidebar.number_input("Credits Completed", min_value=0, max_value=200, value=40)

# Example categorical feature
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())

# --------------------------------------------------
# 6. Create input dataframe
# --------------------------------------------------
input_dict = {
    "Age": age,
    "Credits_Completed": credits,
    "Gender": gender
}

input_df = pd.DataFrame([input_dict])

# Encode input same as training data
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

input_scaled = scaler.transform(input_encoded)

# --------------------------------------------------
# 7. Predict GPA
# --------------------------------------------------
if st.button("🎯 Predict GPA"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted GPA: **{prediction:.2f}**")

# --------------------------------------------------
# 8. Feature importance
# --------------------------------------------------
st.subheader("🔍 Feature Importance")

coef_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Coefficient": model.coef_
}).sort_values("Coefficient", key=np.abs, ascending=False)

st.dataframe(coef_df)
