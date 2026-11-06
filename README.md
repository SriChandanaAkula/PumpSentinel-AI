# 🛠⚡ PumpSentinel AI
## Industrial Pump Health Prediction System

🔮 Powered by Machine Learning • 📊 Predictive Maintenance • 🏭 Industrial Intelligence

---

## 🚀 Overview

**PumpSentinel AI** is a Machine Learning–based industrial monitoring system that predicts the health of industrial pumps using three core parameters:

- 💠 Vibration (mm/s)
- 🔥 Temperature (°C)
- ⚡ Motor Current (A)

### Pump Health Status:
- 🟢 HEALTHY  
- 🟠 WARNING  
- 🔴 CRITICAL  

This system helps industries reduce downtime, prevent failures, and improve operational safety using AI-driven insights.

---

## 🌟 Features

- 🧠 ML-powered pump health classification
- 📈 Failure risk probability score
- 📊 Prediction using vibration, temperature, and current
- 🛠 Actionable maintenance recommendations
- 🎛 Clean & modern Streamlit UI
- ⚡ Lightweight and fast
- 🔒 No external API dependency
- 🌐 Works fully offline
- 🚀 Deployable on Streamlit Cloud

---

## 🧠 How It Works

### 1️⃣ Model Training

The system uses **RandomForestClassifier** trained on historical pump data.

**Input Features**
- Vibration
- Temperature
- Motor Current

**Output Labels**
- HEALTHY
- WARNING
- FAIL

Run training:
```bash
python train_model.py --csv data/pumphealth.csv --out model
```

Generated files:
- pump_model.pkl
- scaler.pkl
- feature_meta.json

---

### 2️⃣ Run the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

Enter pump parameters to get:
- Health Status
- Failure Risk Score
- Maintenance Suggestions

---

## 📦 Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
- streamlit
- scikit-learn
- pandas
- numpy
- joblib

✔ No internet required  
✔ Zero API cost  

---

## 📁 Project Structure

```
PumpSentinel-AI/
│── app.py
│── train_model.py
│── requirements.txt
│── data/
│     └── pumphealth.csv
│── model/
│     ├── pump_model.pkl
│     ├── scaler.pkl
│     └── feature_meta.json
│── README.md
```

---

## 🎨 UI Highlights

- Minimal & fast interface
- Easy numeric inputs
- Color-coded output (Green / Orange / Red)
- Clear maintenance guidance
- Industry-ready design

---

## 🚀 Deployment (Streamlit Cloud)

1. Push the project to GitHub  
2. Go to https://streamlit.io/cloud  
3. Select your repository  
4. Click **Deploy**
5. Project live link: https://pumpsentinel-ai-dytfkp4iqb6etnrtinvp8j.streamlit.app/

---

## 🏆 Why PumpSentinel AI?

- 🔮 Predict failures early
- 🛠 Reduce maintenance costs
- ⚙ Improve pump reliability
- 🧠 AI-driven smart maintenance
- 🌐 Fully offline support

---

## 👨‍💻 Developer

**Tenet Σ**  
ML • Data Science • AI Systems • Industrial Automation
