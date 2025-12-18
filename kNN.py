import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KNN Social Ads", layout="centered")
st.title("📊 Social Network Ads – KNN Classifier")

uploaded = st.file_uploader("Upload Social_Network_Ads.csv", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    X = data[["Age", "EstimatedSalary"]]
    y = data["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    k = st.slider("Select K (Neighbors)", 1, 25, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)

    accuracy = model.score(X_test_s, y_test) * 100
    st.success(f"Model Accuracy: {accuracy:.2f}%")

    st.subheader("🔮 Predict New User")
    age = st.number_input("Age", 18, 70, 30)
    salary = st.number_input("Estimated Salary", 15000, 200000, 50000)

    if st.button("Predict Purchase"):
        sample = scaler.transform([[age, salary]])
        result = model.predict(sample)[0]
        st.info("Purchased" if result == 1 else "Not Purchased")

    st.subheader("📈 Decision Boundary")
    xx, yy = np.meshgrid(
        np.arange(X["Age"].min()-1, X["Age"].max()+1, 0.5),
        np.arange(X["EstimatedSalary"].min()-5000, X["EstimatedSalary"].max()+5000, 3000)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_s = scaler.transform(grid)
    Z = model.predict(grid_s).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X["Age"], X["EstimatedSalary"], c=y)
    ax.set_xlabel("Age")
    ax.set_ylabel("Estimated Salary")
    st.pyplot(fig)

else:
    st.warning("Upload CSV file to start")
