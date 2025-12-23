import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")

# --------------------------------------------------
# 1. Upload CSV file
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Titanic Dataset CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Survived" not in df.columns:
        st.error("CSV file must contain 'Survived' column")
        st.stop()

    # --------------------------------------------------
    # 2. Data Cleaning
    # --------------------------------------------------
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Drop unnecessary columns if present
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --------------------------------------------------
    # 3. Prepare data
    # --------------------------------------------------
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_ss = scaler.fit_transform(X_train)
    X_test_ss = scaler.transform(X_test)

    # --------------------------------------------------
    # 4. Train Logistic Regression model
    # --------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_ss, y_train)

    # --------------------------------------------------
    # 5. Model Evaluation
    # --------------------------------------------------
    y_pred = model.predict(X_test_ss)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**Accuracy:** {accuracy:.4f}")

    st.subheader("📌 Confusion Matrix")
    st.write(cm)

    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # --------------------------------------------------
    # 📈 GRAPH 1: Confusion Matrix Plot
    # --------------------------------------------------
    st.subheader("📈 Confusion Matrix Visualization")

    fig1, ax1 = plt.subplots()
    ax1.imshow(cm)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig1)

    # --------------------------------------------------
    # 6. Sidebar Input
    # --------------------------------------------------
    st.sidebar.header("Passenger Details")

    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=25)
    sibsp = st.sidebar.number_input("Siblings/Spouses", min_value=0, max_value=10, value=0)
    parch = st.sidebar.number_input("Parents/Children", min_value=0, max_value=10, value=0)
    fare = st.sidebar.number_input("Fare", min_value=0.0, value=32.0)
    embarked = st.sidebar.selectbox("Embarked", ["S", "C", "Q"])

    # --------------------------------------------------
    # 7. Create input dataframe
    # --------------------------------------------------
    input_data = {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex": sex,
        "Embarked": embarked
    }

    input_df = pd.DataFrame([input_data])

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    # --------------------------------------------------
    # 8. Predict Survival
    # --------------------------------------------------
    if st.button("🎯 Predict Survival"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"Survived ✅ (Probability: {probability:.2f})")
        else:
            st.error(f"Not Survived ❌ (Probability: {probability:.2f})")

    # --------------------------------------------------
    # 📊 GRAPH 2: Feature Importance
    # --------------------------------------------------
    st.subheader("📊 Feature Importance")

    coef_df = pd.DataFrame({
        "Feature": X_encoded.columns,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=np.abs, ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(coef_df["Feature"][:10], coef_df["Coefficient"][:10])
    ax2.set_title("Top 10 Important Features")
    ax2.invert_yaxis()

    st.pyplot(fig2)

    st.subheader("📋 Feature Coefficients Table")
    st.dataframe(coef_df)

else:
    st.info("👆 Upload Titanic CSV file to continue")
