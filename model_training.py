import os
import logging
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from xgboost import XGBClassifier
import pyttsx3
import speech_recognition as sr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


HEALTH_FILE = os.path.join(".", "health_monitoring.csv")
SAFETY_FILE = os.path.join(".", "safety_monitoring.csv")
REMINDERS_FILE = os.path.join(".", "daily_reminder.csv")


def llama_infer(prompt: str, timeout: int = 30) -> str:
    try:
        
        status_check = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if status_check.returncode != 0:
            return "Error: Ollama service is not running. Please start Ollama first."
        
        result = subprocess.run(
            ["ollama", "run", "tinyllama", prompt],  
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        
        if result.stderr:
            logging.warning(f"Ollama warning: {result.stderr}")
        return result.stdout.strip() or "No response generated"
        
    except subprocess.TimeoutExpired:
        return generate_fallback_summary(prompt)
    except subprocess.CalledProcessError as e:
        logging.error(f"Ollama process error: {e}")
        return generate_fallback_summary(prompt)
    except FileNotFoundError:
        return "Error: Ollama not found. Please install Ollama from https://ollama.ai"
    except Exception as e:
        logging.error(f"Unexpected error running TinyLlama inference: {e}")
        return generate_fallback_summary(prompt)

def generate_fallback_summary(prompt: str) -> str:
    """Generate a basic summary when LLM fails"""
    if "Summarize the performance" in prompt:
        
        import re
        health_clean = re.search(r"Clean Holdout Accuracy: ([\d.]+)", prompt)
        health_noisy = re.search(r"Noisy Holdout Accuracy: ([\d.]+)", prompt)
        fall_clean = re.search(r"Fall Detection Model \(Clean Holdout Accuracy: ([\d.]+)", prompt)
        fall_noisy = re.search(r"Noisy Holdout Accuracy: ([\d.]+)", prompt)
        
        summary = [
            "Model Performance Summary (Automated Fallback):",
            "",
            "Health Alert Model:",
            f"- Clean Data Accuracy: {health_clean.group(1) if health_clean else 'N/A'}",
            f"- Noisy Data Accuracy: {health_noisy.group(1) if health_noisy else 'N/A'}",
            "",
            "Fall Detection Model:",
            f"- Clean Data Accuracy: {fall_clean.group(1) if fall_clean else 'N/A'}",
            f"- Noisy Data Accuracy: {fall_noisy.group(1) if fall_noisy else 'N/A'}",
            "",
            "Recommendations:",
            "1. Consider model deployment with regular performance monitoring",
            "2. Implement gradual rollout strategy",
            "3. Set up automated retraining pipeline",
            "4. Monitor system resource usage in production"
        ]
        return "\n".join(summary)
    return "Automated summary not available for this prompt type"


def preprocess_health(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    if 'Blood Pressure' in df.columns:
        bp_extracted = df['Blood Pressure'].str.extract(r'(\d+)/(\d+)')
        df['Systolic_BP'] = pd.to_numeric(bp_extracted[0], errors='coerce')
        df['Diastolic_BP'] = pd.to_numeric(bp_extracted[1], errors='coerce')
        df.drop(columns=['Blood Pressure'], inplace=True)
    yes_no_cols = [
        'Heart Rate Below/Above Threshold (Yes/No)',
        'Blood Pressure Below/Above Threshold (Yes/No)',
        'Glucose Levels Below/Above Threshold (Yes/No)',
        'SpO₂ Below Threshold (Yes/No)',
        'Alert Triggered (Yes/No)',
        'Caregiver Notified (Yes/No)'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df.loc[:, col] = (df[col].replace({'Yes': 1, 'No': 0})
                            .infer_objects(copy=False)
                            .astype(int))
    if 'Glucose Levels' in df.columns:
        df.loc[:, 'Glucose Levels'] = df['Glucose Levels'].replace('-', pd.NA)
        df.loc[:, 'Glucose Levels'] = pd.to_numeric(df['Glucose Levels'], errors='coerce')
    df = df.dropna(subset=['Heart Rate', 'Systolic_BP', 'Diastolic_BP'], how='any').copy()
    return df


def preprocess_safety(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    binary_cols = ['Fall Detected (Yes/No)', 'Alert Triggered (Yes/No)', 'Caregiver Notified (Yes/No)']
    for col in binary_cols:
        if col in df.columns:
            df.loc[:, col] = (df[col].replace({'Yes': 1, 'No': 0})
                            .infer_objects(copy=False)
                            .astype(int))
    if 'Movement Activity' in df.columns:
        df.loc[:, 'Movement Activity Encoded'] = LabelEncoder().fit_transform(df['Movement Activity'].astype(str))
    if 'Impact Force Level' in df.columns:
        df.loc[:, 'Impact Force Level Encoded'] = LabelEncoder().fit_transform(df['Impact Force Level'].astype(str))
    df = df.dropna(subset=['Movement Activity Encoded', 'Impact Force Level Encoded', 'Post-Fall Inactivity Duration (Seconds)']).copy()
    return df


def preprocess_reminders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    for col in ['Reminder Sent (Yes/No)', 'Acknowledged (Yes/No)']:
        if col in df.columns:
            df.loc[:, col] = (df[col].replace({'Yes': 1, 'No': 0})
                            .infer_objects(copy=False)
                            .astype(int))
    return df



def train_xgboost_model(X: pd.DataFrame, y: pd.Series, model_name: str = "Model"):
    
    if X.isnull().any().any():
        logging.warning(f"{model_name}: Input contains null values. Filling with median.")
        X = X.fillna(X.median())
    
    if not X.shape[0] == y.shape[0]:
        raise ValueError(f"{model_name}: Feature and target dimensions don't match")
    
    if y.isnull().any():
        raise ValueError(f"{model_name}: Target variable contains null values")
    
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    
 
    param_dist = {
        "n_estimators": [30, 50, 70],
        "max_depth": [1, 2],
        "learning_rate": [0.001, 0.005, 0.01],
        "subsample": [0.4, 0.5],
        "colsample_bytree": [0.4, 0.5],
        "min_child_weight": [5, 7, 9],
        "gamma": [0.5, 0.7],
        "reg_alpha": [1.0, 2.0],
        "reg_lambda": [3.0, 4.0]
    }
    
    xgb_model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        scale_pos_weight=len(y[y==0])/len(y[y==1])
    )
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_model, param_distributions=param_dist, n_iter=10, cv=cv,
        scoring='accuracy', verbose=1, random_state=42, n_jobs=-1
    )
    
    
    search.fit(X_train, y_train)
    best_params = search.best_params_
    
    
    final_model = XGBClassifier(
        **best_params,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        scale_pos_weight=len(y[y==0])/len(y[y==1]),
        early_stopping_rounds=5
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds_test = final_model.predict(X_val)
    acc_test = accuracy_score(y_val, preds_test)
    logging.info(f"{model_name} - Internal Test Accuracy: {acc_test:.4f}")
    logging.info(f"{model_name} - Internal Test Report:\n{classification_report(y_val, preds_test)}")
    
    return final_model, X_holdout, y_holdout


pd.set_option('future.no_silent_downcasting', True)


def add_noise(X: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
    X_noisy = X.copy()
    numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        std = X_noisy[col].std()
        if std > 0:  
            noise = np.random.normal(0, noise_level * std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col].astype('float64') + noise
    
    return X_noisy


def display_reminders(df: pd.DataFrame):
    if df.empty:
        logging.info("No reminder data available.")
        return None, None, None
    
    
    total_reminders = len(df)
    acknowledged = df['Acknowledged (Yes/No)'].sum()
    missed = total_reminders - acknowledged
    
    
    df['Hour'] = pd.to_datetime(df['Scheduled Time'], format='%H:%M:%S').dt.hour.astype(int)
    hourly_ack_rates = df.groupby('Hour')['Acknowledged (Yes/No)'].mean() * 100
    best_hours = hourly_ack_rates.sort_values(ascending=False).head(3)
    
  
    reminder_types = df['Reminder Type'].value_counts()
    type_ack_rates = df.groupby('Reminder Type')['Acknowledged (Yes/No)'].mean() * 100
    
    logging.info("\nReminder Statistics:")
    logging.info(f"Total Reminders: {total_reminders}")
    logging.info(f"Acknowledged: {acknowledged} ({acknowledged/total_reminders*100:.1f}%)")
    logging.info(f"Missed: {missed} ({missed/total_reminders*100:.1f}%)")
    
    logging.info("\nReminder Type Distribution:")
    for rtype, count in reminder_types.items():
        ack_rate = type_ack_rates[rtype]
        logging.info(f"{rtype}: {count} reminders, {ack_rate:.1f}% acknowledged")
    
    logging.info("\nBest Acknowledgment Hours:")
    best_hours = hourly_ack_rates.sort_values(ascending=False).head(3)
    for hour, rate in best_hours.items():
        logging.info(f"{hour:02d}:00 - {rate:.1f}% acknowledged")
    
    logging.info("\nWorst Acknowledgment Hours:")
    worst_hours = hourly_ack_rates.sort_values().head(3)
    for hour, rate in worst_hours.items():
        logging.info(f"{hour:02d}:00 - {rate:.1f}% acknowledged")
    
    logging.info("\nDaily Reminders:")
    for _, row in df.iterrows():
        reminder_type = row.get('Reminder Type', 'Unknown Reminder')
        scheduled_time = row.get('Scheduled Time', 'Unknown Time')
        user_id = row.get('Device-ID/User-ID', 'Unknown User')
        acknowledged = row.get('Acknowledged (Yes/No)', 0)
        status = "ACKNOWLEDGED" if acknowledged == 1 else "MISSED"
        logging.info(f"{reminder_type} at {scheduled_time} for {user_id} => {status}")

def simple_assistant_chat(health_metrics: dict, fall_metrics: dict, reminder_stats: dict) -> None:
    print("\nVoice-Enabled Health Assistant Ready (Say 'goodbye' to exit)")
    
    engine = pyttsx3.init()
    recognizer = sr.Recognizer()
    
    engine.setProperty('rate', 145)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    def speak(text):
        print(f"\nAssistant: {text}")
        engine.say(text)
        engine.runAndWait()
    
    def listen():
        with sr.Microphone() as source:
            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                user_input = recognizer.recognize_google(audio).lower()
                print(f"You: {user_input}")
                return user_input
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                return ""
            except Exception as e:
                logging.error(f"Error in speech recognition: {e}")
                return ""
    
    speak("Hello! I'm your health assistant. How can I help you today?")
    
    while True:
        try:
            user_input = listen()
            
            if not user_input:
                speak("I didn't catch that. Could you please repeat?")
                continue
            
            if "goodbye" in user_input or "bye" in user_input:
                speak("Take care and stay healthy! Goodbye!")
                break
            
            context = f"""You are a helpful and friendly health assistant for elderly people.
Current health monitoring accuracy: {health_metrics['clean_acc']*100:.1f}%
Fall detection accuracy: {fall_metrics['clean_acc']*100:.1f}%
Best reminder time: {reminder_stats['best_hour']:02d}:00
Overall reminder acknowledgment rate: {reminder_stats['ack_percent']:.1f}%

User Query: {user_input}"""
            
            response = llama_infer(context)
            speak(response)
            
        except KeyboardInterrupt:
            speak("Goodbye! Take care!")
            break
        except Exception as e:
            logging.error(f"Error in assistant chat: {e}")
            speak("I'm having trouble understanding. Could you please repeat?")

def main():
    acc_holdout_health = acc_holdout_health_noisy = acc_holdout_fall = acc_holdout_fall_noisy = None
    
    try:
        health_df = pd.read_csv(HEALTH_FILE)
        safety_df = pd.read_csv(SAFETY_FILE)
        reminders_df = pd.read_csv(REMINDERS_FILE)
        health_df = preprocess_health(health_df)
        safety_df = preprocess_safety(safety_df)
        reminders_df = preprocess_reminders(reminders_df)
    except FileNotFoundError as e:
        logging.error(f"Error loading data: {e} - Please ensure the CSV files are in the correct directory.")
        return
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        return

    try:
        X_health = health_df[['Heart Rate', 'Systolic_BP', 'Diastolic_BP', 'Glucose Levels', 
                             'SpO₂ Below Threshold (Yes/No)', 'Heart Rate Below/Above Threshold (Yes/No)',
                             'Blood Pressure Below/Above Threshold (Yes/No)', 'Glucose Levels Below/Above Threshold (Yes/No)']].copy()
        
        binary_cols = ['SpO₂ Below Threshold (Yes/No)', 'Heart Rate Below/Above Threshold (Yes/No)',
                      'Blood Pressure Below/Above Threshold (Yes/No)', 'Glucose Levels Below/Above Threshold (Yes/No)']
        X_health[binary_cols] = X_health[binary_cols].astype(int)
        
        numeric_cols = ['Heart Rate', 'Systolic_BP', 'Diastolic_BP', 'Glucose Levels']
        X_health[numeric_cols] = X_health[numeric_cols].astype('float64')
        X_health = X_health.fillna(X_health.median())
        y_health = health_df['Alert Triggered (Yes/No)'].astype(int)
        
        if not y_health.isnull().any() and y_health.nunique() >= 2:
            clf_health, X_holdout_health, y_holdout_health = train_xgboost_model(X_health, y_health, "Health Alert Model")
            preds_holdout_health = clf_health.predict(X_holdout_health)
            acc_holdout_health = accuracy_score(y_holdout_health, preds_holdout_health)
            X_holdout_health_noisy = add_noise(X_holdout_health)
            preds_holdout_health_noisy = clf_health.predict(X_holdout_health_noisy)
            acc_holdout_health_noisy = accuracy_score(y_holdout_health, preds_holdout_health_noisy)
            joblib.dump(clf_health, "health_alert_xgb_model.pkl")
    except Exception as e:
        logging.error(f"Error in Health Alert Model: {e}")

    try:
        X_fall = safety_df[['Movement Activity Encoded', 'Impact Force Level Encoded', 'Post-Fall Inactivity Duration (Seconds)']]
        y_fall = safety_df['Fall Detected (Yes/No)'].astype(int)
        
        if y_fall.nunique() >= 2:
            clf_fall, X_holdout_fall, y_holdout_fall = train_xgboost_model(X_fall, y_fall, "Fall Detection Model")
            preds_holdout_fall = clf_fall.predict(X_holdout_fall)
            acc_holdout_fall = accuracy_score(y_holdout_fall, preds_holdout_fall)
            X_holdout_fall_noisy = add_noise(X_holdout_fall)
            preds_holdout_fall_noisy = clf_fall.predict(X_holdout_fall_noisy)
            acc_holdout_fall_noisy = accuracy_score(y_holdout_fall, preds_holdout_fall_noisy)
            joblib.dump(clf_fall, "fall_detection_xgb_model.pkl")
    except Exception as e:
        logging.error(f"Error in Fall Detection Model: {e}")

    reminder_data = display_reminders(reminders_df)
    
    try:
        health_metrics = {'clean_acc': acc_holdout_health or 0.0, 'noisy_acc': acc_holdout_health_noisy or 0.0}
        fall_metrics = {'clean_acc': acc_holdout_fall or 0.0, 'noisy_acc': acc_holdout_fall_noisy or 0.0}
        
        if reminder_data:
            acknowledged, total_reminders, best_hours = reminder_data
            reminder_stats = {
                'ack_percent': (acknowledged/total_reminders*100) if total_reminders > 0 else 0.0,
                'best_hour': int(best_hours.index[0]) if not best_hours.empty else 0
            }
        else:
            reminder_stats = {'ack_percent': 0.0, 'best_hour': 0}
        
        simple_assistant_chat(health_metrics, fall_metrics, reminder_stats)
    except Exception as e:
        logging.error(f"Error in assistant chat: {e}")
    
    logging.info("System update complete!")

if __name__ == "__main__":
    main()