import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# LOAD MODEL + FEATURES
# -------------------------------
with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Digital Twin", layout="centered")

st.title("🧠 AI Digital Twin System")
st.markdown("### Predict behavior & get smart recommendations")

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("📥 Enter Your Current State")

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of Day", 0, 23, 10)
    day_of_week = st.selectbox("Day of Week", list(range(7)))
    sleep_hours = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
    energy_level = st.slider("Energy Level", 0.0, 1.0, 0.5)
    stress_level = st.slider("Stress Level", 0.0, 1.0, 0.5)

with col2:
    phone_usage = st.slider("Phone Usage (minutes)", 0, 300, 100)
    social_interactions = st.slider("Social Interactions", 0, 50, 10)
    study_hours = st.slider("Study Hours", 0.0, 6.0, 2.0)
    workload = st.slider("Workload", 0.0, 10.0, 5.0)
    mood = st.slider("Mood", 0.0, 2.0, 1.0)

previous_activity = st.selectbox(
    "Previous Activity (0=exercise, 1=scroll, 2=sleep, 3=study)",
    [0, 1, 2, 3]
)

st.divider()

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict Behavior"):

    # Create input dictionary
    input_data = {
        "hour": hour,
        "day_of_week": day_of_week,
        "sleep_hours": sleep_hours,
        "energy_level": energy_level,
        "stress_level": stress_level,
        "phone_usage": phone_usage,
        "social_interactions": social_interactions,
        "study_hours": study_hours,
        "workload": workload,
        "mood": mood,
        "previous_activity": previous_activity
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure same feature order as training
    input_df = input_df[feature_columns]

    # Prediction
    pred = rf_model.predict(input_df)[0]

    # Map labels (IMPORTANT: same order as training)
    activity_map = {
        0: "exercise",
        1: "scroll",
        2: "sleep",
        3: "study"
    }

    predicted_activity = activity_map.get(pred, "unknown")

    st.subheader(f"📊 Predicted Activity: {predicted_activity}")

    # -------------------------------
    # DECISION ENGINE
    # -------------------------------
    if predicted_activity == "scroll":
        if stress_level > 0.6:
            recommendation = "🧘 Take a short break or meditate"
        else:
            recommendation = "📚 Try studying for 30 minutes"

    elif predicted_activity == "study":
        if energy_level < 0.5:
            recommendation = "😴 Take some rest before studying"
        else:
            recommendation = "🔥 Great time to focus!"

    elif predicted_activity == "sleep":
        recommendation = "😴 Maintain a healthy sleep routine"

    elif predicted_activity == "exercise":
        recommendation = "🏃 Good time for physical activity"

    else:
        recommendation = "No recommendation"

    st.success(f"💡 Recommendation: {recommendation}")

    st.divider()

    # -------------------------------
    # EXPLAINABILITY
    # -------------------------------
    st.subheader("🔍 Why this prediction?")

    importance = rf_model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(df_imp.set_index("Feature"))

    st.info("Top features influencing this prediction")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.caption("🚀 AI Digital Twin | Machine Learning + Decision Intelligence")