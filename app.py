%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")

# Load data and model
merged_data = pd.read_pickle("merged_student_dataset.pkl")
model = joblib.load("student_performance_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_features = joblib.load("model_features.pkl")

st.title("🎓 Student Performance Dashboard")

st.sidebar.header("🔍 Filters & Inputs")
subject_choice = st.sidebar.selectbox("Select Subject", ["Math", "Portuguese"])
roll_no = st.sidebar.text_input("Enter Roll No (or ID)", "")

st.subheader("📊 Attendance Insights")

# Attendance histogram
if "absences_mat" in merged_data.columns and "absences_por" in merged_data.columns:
    merged_data["avg_absences"] = (merged_data["absences_mat"] + merged_data["absences_por"]) / 2
    fig, ax = plt.subplots()
    ax.hist(merged_data["avg_absences"], bins=10, color="lightblue", edgecolor="black")
    ax.set_title("Attendance Distribution (Absences)")
    ax.set_xlabel("Number of Absences")
    ax.set_ylabel("Number of Students")
    st.pyplot(fig)
else:
    st.warning("Absence columns not found in dataset.")

st.subheader("🎯 Predict Student Performance")

if roll_no:
    # Just a sample (replace with roll-based lookup if you add IDs)
    student_row = merged_data.sample(1)
    st.write(f"Showing prediction for Roll No: {roll_no} (sample data)")

    X_input = student_row[model_features].copy()
    for col in X_input.columns:
        if col in label_encoders:
            X_input[col] = label_encoders[col].transform(X_input[col])

    predicted_grade = model.predict(X_input)[0]
    st.metric(label="Predicted Final Grade (G3)", value=f"{predicted_grade:.2f}")

    fig, ax = plt.subplots()
    ax.bar(["Predicted G3"], [predicted_grade], color="seagreen")
    ax.set_ylim(0, 20)
    st.pyplot(fig)
else:
    st.info("Enter a roll number to predict student performance.")
