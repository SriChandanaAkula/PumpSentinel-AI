
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PumpSentinel AI",
    page_icon="🛠️",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------
MODEL_DIR = "model"

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "pump_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_meta.json"), "r") as f:
        meta = json.load(f)

    return model, scaler, meta

try:
    model, scaler, meta = load_model()
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()

label_map = meta["label_map"]
inv_label_map = {int(v): k for k, v in label_map.items()}

# ---------------------------------------------------------
# AI HYPOTHESIS
# ---------------------------------------------------------
def generate_hypothesis(vibration, temperature, power, status, risk):

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return """
**General diagnostic guidance**

- Review vibration trends for possible mechanical abnormalities.
- Check temperature trends for possible thermal abnormalities.
- Review electrical power consumption for unusual changes.

These are general checks, not confirmed failure causes.
"""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are an industrial pump monitoring assistant.

Dataset-derived sensor features:
- Vibration feature: {vibration}
- Temperature feature: {temperature}
- Electrical power feature: {power}

ML predicted status: {status}
Predicted critical-class probability: {risk:.3f}

Explain the prediction cautiously.
Provide:
1. Possible interpretation
2. Three maintenance checks
3. Suggested urgency

Do not claim a confirmed mechanical fault from these features alone.
The values are dataset-derived features, not necessarily calibrated
physical measurements.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You assist with industrial pump condition monitoring."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception:
        return """
AI hypothesis service is currently unavailable.

Please review the sensor trends and follow the recommended
maintenance inspection procedures.
"""

# ---------------------------------------------------------
# SESSION HISTORY
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

def clear_history():
    st.session_state.history = []

# ---------------------------------------------------------
# APPLICATION UI
# ---------------------------------------------------------
st.title("🛠️ PumpSentinel AI")
st.markdown(
    "### Intelligent Industrial Pump Health Monitoring"
)

st.write(
    "Enter the three dataset-derived sensor features to "
    "predict the pump's operating condition."
)

col1, col2, col3 = st.columns(3)

vibration = col1.number_input(
    "Vibration feature (VS1)",
    value=3.0,
    step=0.1
)

temperature = col2.number_input(
    "Temperature feature (TS1–TS4)",
    value=35.0,
    step=0.1
)

power = col3.number_input(
    "Electrical power feature (EPS1)",
    value=6.0,
    step=0.1
)

analyze = st.button("🔍 Analyze Pump Health")

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if analyze:

    try:
        features = meta["features"]

        input_values = {
            "vibration": vibration,
            "temperature": temperature,
            "power": power
        }

        X = np.array(
            [[input_values[name] for name in features]],
            dtype=float
        )

        X_scaled = scaler.transform(X)

        probabilities = model.predict_proba(X_scaled)[0]

        predicted_class = int(model.predict(X_scaled)[0])
        status = inv_label_map[predicted_class]

        class_to_probability = {
            int(class_id): float(probability)
            for class_id, probability
            in zip(model.classes_, probabilities)
        }

        risk = class_to_probability.get(
            int(label_map["CRITICAL"]), 0.0
        )

        st.subheader("📊 Prediction Result")

        st.metric("Pump Health Status", status)
        st.metric("Critical-Class Probability", f"{risk * 100:.2f}%")

        st.caption(
            "This probability represents the model's estimated "
            "probability of the CRITICAL class, not a calibrated "
            "real-world failure probability."
        )

        # AI explanation
        st.subheader("🤖 AI Diagnostic Hypothesis")

        hypothesis = generate_hypothesis(
            vibration, temperature, power, status, risk
        )

        st.write(hypothesis)

        # General maintenance guidance
        st.subheader("🔧 Maintenance Recommendations")

        if status == "CRITICAL":
            st.error(
                "Prioritize inspection and follow your plant's "
                "safety and maintenance procedures."
            )
        elif status == "WARNING":
            st.warning(
                "Review sensor trends and schedule a maintenance inspection."
            )
        else:
            st.success(
                "Continue routine monitoring and preventive maintenance."
            )

        st.write("""
        - Inspect pump and motor condition.
        - Review vibration, temperature and power trends.
        - Compare readings with equipment-specific operating limits.
        """)

        # Store history
        st.session_state.history.append({
            "timestamp": int(time.time()),
            "vibration": vibration,
            "temperature": temperature,
            "power": power,
            "status": status,
            "risk": risk,
            "hypothesis": hypothesis
        })

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
st.header("📈 Pump Monitoring Dashboard")

if not st.session_state.history:

    st.info("Perform an analysis to populate the dashboard.")

else:

    import altair as alt

    df = pd.DataFrame(st.session_state.history)
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")

    st.subheader("Vibration vs Critical-Class Probability")

    chart1 = alt.Chart(df).mark_circle(size=90).encode(
        x="vibration:Q",
        y="risk:Q",
        color="status:N",
        tooltip=["time", "vibration", "temperature", "power", "status", "risk"]
    ).interactive()

    st.altair_chart(chart1, use_container_width=True)

    st.subheader("Critical-Class Probability Over Time")

    chart2 = alt.Chart(df).mark_line(
        point=True
    ).encode(
        x="time:T",
        y="risk:Q",
        color="status:N"
    ).interactive()

    st.altair_chart(chart2, use_container_width=True)

    st.subheader("Analysis History")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download History CSV",
        df.to_csv(index=False).encode("utf-8"),
        "pumpsentinel_history.csv",
        "text/csv"
    )

    if st.button("Clear History"):
        clear_history()
        st.rerun()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("About PumpSentinel AI")
    st.write("ML-based industrial pump condition monitoring.")
    st.write("Model: Random Forest Classifier")
    st.write("Dataset: UCI Hydraulic Systems")
