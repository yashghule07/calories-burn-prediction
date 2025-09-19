# app.py
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, encoder, scaler
model = load("model.joblib")
encoder = load("encoder.joblib")
scaler = load("scaler.joblib")

# Load dataset for dashboard
df = pd.read_csv("activity_dataset.csv")

# Streamlit config
st.set_page_config(page_title="Calories Burn Prediction", layout="wide")

# Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Prediction"])

# -------------------------
# DASHBOARD TAB
# -------------------------
with tab1:
    st.title("📊 Calories Burn Dashboard")

    # KPIs
    total_activities = df["Activity"].nunique()
    avg_calories = df["Calories"].mean()
    max_calories = df["Calories"].max()

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Unique Activities", total_activities)
    kpi2.metric("Avg Calories Burned", f"{avg_calories:.2f}")
    kpi3.metric("Max Calories Burned", f"{max_calories:.2f}")

    st.markdown("---")

    # Visualization 1: Activity distribution
    st.subheader("Activity Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Activity", data=df, palette="viridis", ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Visualization 2: Avg calories per activity
    st.subheader("Average Calories per Activity")
    avg_cal = df.groupby("Activity")["Calories"].mean().sort_values()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    avg_cal.plot(kind="bar", color="teal", ax=ax2)
    plt.ylabel("Avg Calories")
    st.pyplot(fig2)

    # Visualization 3: Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # Visualization 4: Calories contribution pie chart
    st.subheader("Calories Contribution by Activity")
    calories_sum = df.groupby("Activity")["Calories"].sum()
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    ax4.pie(calories_sum, labels=calories_sum.index, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("Set2"))
    ax4.axis("equal")
    st.pyplot(fig4)

    # Show actual vs predicted plot
    st.subheader("Actual vs Predicted Calories")
    st.image("actual_vs_predicted.png", caption="Model Predictions")

# -------------------------
# PREDICTION TAB
# -------------------------
with tab2:
    st.title("🤖 Calories Burn Prediction")

    # Sidebar input
    st.sidebar.header("Input Parameters")
    activity = st.sidebar.selectbox("Activity", df["Activity"].unique())
    duration = st.sidebar.slider("Duration (minutes)", 1, 180, 30)
    speed = st.sidebar.slider("Speed (km/h)", 0.0, 30.0, 5.0)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)

    if st.sidebar.button("Predict"):
        cat_encoded = encoder.transform([[activity]])
        num_scaled = scaler.transform([[duration, speed, weight]])
        input_final = np.hstack([num_scaled, cat_encoded])
        prediction = model.predict(input_final)[0]
        st.success(f"🔥 Estimated Calories Burned: **{prediction:.2f} kcal**")

    # Batch prediction
    st.header("📂 Upload CSV for Batch Predictions")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        cat_encoded = encoder.transform(data[["Activity"]])
        num_scaled = scaler.transform(data[["Duration_min", "Speed_kmph", "Weight_kg"]])
        final_input = np.hstack([num_scaled, cat_encoded])

        preds = model.predict(final_input)
        data["Predicted_Calories"] = preds
        st.dataframe(data.head(20))

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
