# Khayal 🧓🏽🩺 – Elderly Health and Safety Monitor

Khayal is an intelligent AI-based assistant designed to monitor the **health**, **safety**, and **daily needs** of elderly individuals. It uses machine learning for fall detection, health anomaly prediction, and supports daily reminder scheduling — ensuring peace of mind for caregivers and independence for the elderly.

## 💡 Features

- ✅ **Fall Detection** using XGBoost  
- ✅ **Health Monitoring** for early warning signs  
- ✅ **Daily Reminders** for medication, tasks, and appointments  
- ✅ Simple CSV-based interface for input and output  
- ✅ Easy to extend for real-world smart home integrations  

## 🗂️ Project Structure

    ├── model_training.py                # Script to train health and fall models  
    ├── fall_detection_xgb_model.pkl    # Pre-trained fall detection model  
    ├── health_alert_xgb_model.pkl      # Pre-trained health alert model  
    ├── health_monitoring.csv           # Sample health data  
    ├── safety_monitoring.csv           # Sample safety data  
    ├── daily_reminder.csv              # Sample daily reminder data  
    └── README.md                       # You are here!  

## ⚙️ Getting Started

### 1. Clone the repository

    git clone https://github.com/yashigauri/Khayal.git  
    cd Khayal  

### 2. Install dependencies

    pip install pandas numpy xgboost  

### 3. (Optional) Train the models

If you want to retrain the models using your own dataset:

    python model_training.py  

This will output updated `.pkl` model files.

## 🧪 Usage

You can integrate the `.pkl` models into your Python-based monitoring or IoT system to:

- Detect falls using `fall_detection_xgb_model.pkl`  
- Monitor health conditions using `health_alert_xgb_model.pkl`  
- Schedule and retrieve reminders from `daily_reminder.csv`  

## 📊 Sample Data

- `health_monitoring.csv`: Heart rate, blood pressure, etc.  
- `safety_monitoring.csv`: Movement, activity logs  
- `daily_reminder.csv`: Tasks like medication, appointments  



