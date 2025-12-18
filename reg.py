import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student GPA Predictor", layout="centered")

st.title("🎓 Student GPA Predictor")

# --------------------------------------------------
# 1. Upload CSV file
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Student Details CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "GPA" not in df.columns:
        st.error("CSV file must contain 'GPA' column")
        st.stop()

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
    # 📈 GRAPH 1: Actual vs Predicted GPA
    # --------------------------------------------------
    st.subheader("📈 Actual vs Predicted GPA")

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred)
    ax1.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()]
    )
    ax1.set_xlabel("Actual GPA")
    ax1.set_ylabel("Predicted GPA")
    ax1.set_title("Actual vs Predicted GPA")
    ax1.grid(True)

    st.pyplot(fig1)

    # --------------------------------------------------
    # 5. Sidebar: User input
    # --------------------------------------------------
    st.sidebar.header("Enter Student Details")

    age = st.sidebar.number_input(
        "Age", min_value=15, max_value=60, value=20
    )

    credits = st.sidebar.number_input(
        "Credits Completed", min_value=0, max_value=200, value=40
    )

    gender = st.sidebar.selectbox(
        "Gender", df["Gender"].unique()
    )

    # --------------------------------------------------
    # 6. Create input dataframe
    # --------------------------------------------------
    input_dict = {
        "Age": age,
        "Credits_Completed": credits,
        "Gender": gender
    }

    input_df = pd.DataFrame([input_dict])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    # --------------------------------------------------
    # 7. Predict GPA
    # --------------------------------------------------
    if st.button("🎯 Predict GPA"):
        prediction = model.predict(input_scaled)[0]
        st.success(f"Predicted GPA: **{prediction:.2f}**")

    # --------------------------------------------------
    # 📊 GRAPH 2: Feature Importance
    # --------------------------------------------------
    st.subheader("📊 Feature Importance")

    coef_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(coef_df["Feature"][:10], coef_df["Coefficient"][:10])
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Top 10 Important Features")
    ax2.invert_yaxis()

    st.pyplot(fig2)

    st.subheader("📋 Feature Coefficients Table")
    st.dataframe(coef_df)

else:
    st.info("👆 Please upload a CSV file to continue")
