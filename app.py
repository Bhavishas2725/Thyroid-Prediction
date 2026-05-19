from flask import Flask, render_template, request, jsonify
import pandas as pd
from datetime import time
import numpy as np
import joblib
import os

app = Flask(__name__) 

def custom_serializer(obj):
    if isinstance(obj, time):
        return obj.strftime('%H:%M:%S')
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Load Recurrence Model
recurrence_model = joblib.load('model/thyroid_xgb_recur.pkl')

# Load Risk Model
risk_model = None
if os.path.exists('model/logistic_risk_model.pkl'):
    try:
        risk_model = joblib.load('model/logistic_risk_model.pkl')
    except Exception as e:
        print(f"Error loading risk model: {e}")
else:
    print("Warning: Risk model file not found. Risk predictions will use a placeholder.")

# Load Label Encoders
label_encoders = joblib.load('model/label_encoders_recur.pkl')

# Load Age Scaler
scaler = joblib.load('model/age_scaler_risk.pkl')

# Load Feature Order for Recurrence
feature_order_recur = joblib.load('model/feature_order_recur.pkl')

# Load Feature Order for Risk
feature_order_risk = None
if os.path.exists('model/feature_order_risk.pkl'):
    try:
        feature_order_risk = joblib.load('model/feature_order_risk.pkl')
    except Exception as e:
        print(f"Error loading risk feature order: {e}")
else:
    print("Warning: Risk feature order file not found.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Collect input from form - required fields for recurrence
        custom_input = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Smoking': request.form['smoking'],
            'Hx Smoking': request.form['hx_smoking'],
            'Hx Radiothreapy': request.form['hx_radiotherapy'],
            'Thyroid Function': request.form['thyroid_function'],
            'Physical Examination': request.form['physical_examination'],
            'Adenopathy': request.form['adenopathy'],
            'Pathology': request.form['pathology'],
            'Focality': request.form['focality'],
            'Stage': request.form['stage'],
            'Risk': request.form['risk'],
            'Response': request.form['response'],
        }

        df = pd.DataFrame([custom_input])
        
        # Encode categorical features
        for col in df.select_dtypes(include=['object']).columns:
            if isinstance(label_encoders, dict) and col in label_encoders:
                try:
                    encoder = label_encoders[col]
                    # Check if encoder is fitted (has classes_ attribute)
                    if hasattr(encoder, 'classes_'):
                        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        df[col] = df[col].map(mapping).fillna(-1).astype(int)
                    else:
                        # Encoder not fitted, try to fit with the value or use unknown encoding
                        unique_values = [df[col].iloc[0]]
                        try:
                            encoder.fit(unique_values)
                            df[col] = encoder.transform(df[col])
                        except:
                            # If fitting fails, encode as integer hash
                            df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
                except Exception as e:
                    print(f"Error encoding column {col}: {e}")
                    # Fallback: encode as integer hash
                    df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
        
        # Scale Age
        if hasattr(scaler, 'transform'):
            try:
                # Use .values to avoid feature names warning
                age_values = df[['Age']].values
                df['Age'] = scaler.transform(age_values).flatten()
            except Exception as e:
                print(f"Error scaling Age: {e}")
        
        # Use feature_order to select columns dynamically
        if not isinstance(feature_order_recur, list):
            feature_order_list = list(feature_order_recur)
        else:
            feature_order_list = feature_order_recur
        
        cols_to_use = [col for col in feature_order_list if col in df.columns]
        df = df[cols_to_use]
        
        # Ensure all columns are numeric (convert any remaining object columns)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
            except:
                pass

        # Make prediction
        try:
            if recurrence_model is not None and hasattr(recurrence_model, 'predict'):
                pred = recurrence_model.predict(df)[0]
                result = 'Yes' if pred == 1 else 'No'
            else:
                print("No valid recurrence model available")
                result = 'Unable to predict: Model not properly trained'
        except Exception as e:
            print(f"Prediction error: {e}")
            result = f'Error: {str(e)}'

        return render_template(
            'index.html',
            prediction_text=f'Recurrence Prediction: {result}'
        )
    else:
        return render_template('index.html')

@app.route('/predict_risk', methods=['POST', 'GET'])
def predict_risk():
    if request.method == 'POST':
        # Collect input from form
        custom_input = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Smoking': request.form['smoking'],
            'Hx Smoking': request.form['hx_smoking'],
            'Hx Radiothreapy': request.form['hx_radiotherapy'],
            'Thyroid Function': request.form['thyroid_function'],
            'Physical Examination': request.form['physical_examination'],
            'Adenopathy': request.form['adenopathy'],
            'Pathology': request.form['pathology'],
            'Focality': request.form['focality'],
            'T': request.form['t_stage'],
            'N': request.form['n_stage'],
            'M': request.form['m_stage'],
            'Stage': request.form['stage'],
        }

        df = pd.DataFrame([custom_input])
        
        # Encode categorical features
        for col in df.select_dtypes(include=['object']).columns:
            if isinstance(label_encoders, dict) and col in label_encoders:
                try:
                    encoder = label_encoders[col]
                    # Check if encoder is fitted (has classes_ attribute)
                    if hasattr(encoder, 'classes_'):
                        mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                        df[col] = df[col].map(mapping).fillna(-1).astype(int)
                    else:
                        # Encoder not fitted, try to fit with the value or use unknown encoding
                        unique_values = [df[col].iloc[0]]
                        try:
                            encoder.fit(unique_values)
                            df[col] = encoder.transform(df[col])
                        except:
                            # If fitting fails, encode as integer hash
                            df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
                except Exception as e:
                    print(f"Error encoding column {col}: {e}")
                    # Fallback: encode as integer hash
                    df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
        
        # Scale Age
        if hasattr(scaler, 'transform'):
            try:
                # Use .values to avoid feature names warning
                age_values = df[['Age']].values
                df['Age'] = scaler.transform(age_values).flatten()
            except Exception as e:
                print(f"Error scaling Age: {e}")
        
        # Use feature_order to select columns
        if feature_order_risk is not None:
            if not isinstance(feature_order_risk, list):
                feature_order_list = list(feature_order_risk)
            else:
                feature_order_list = feature_order_risk
        else:
            # Fallback to recurrence feature order if risk feature order not available
            if not isinstance(feature_order_recur, list):
                feature_order_list = list(feature_order_recur)
            else:
                feature_order_list = feature_order_recur
        
        cols_to_use = [col for col in feature_order_list if col in df.columns]
        df = df[cols_to_use]
        
        # Ensure all columns are numeric (convert any remaining object columns)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
            except:
                pass

        # Make prediction
        try:
            if risk_model is not None and hasattr(risk_model, 'predict'):
                pred = risk_model.predict(df)[0]
                # Assuming multiclass classification (0=Low, 1=High, 2=Intermediate)
                risk_levels = {0: 'Low', 1: 'High', 2: 'Intermediate'}
                result = risk_levels.get(int(pred), str(pred))
            else:
                # Risk model not available - use a placeholder response
                result = 'Risk model not available. Please check model files.'
                print("No valid risk model available - risk model file may be missing")
        except Exception as e:
            print(f"Prediction error in risk prediction: {e}")
            result = f'Prediction Error: {str(e)}'

        return render_template(
            'index.html',
            prediction_text=f'Risk Prediction: {result}'
        )
    else:
        return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').lower()
    
    response = get_chatbot_response(user_message)
    return jsonify({'response': response})

def get_chatbot_response(message):
    # Basic keyword-based response logic
    if any(word in message for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your Thyroid Assistant. How can I help you today?"
    
    if any(word in message for word in ['how', 'help', 'use']):
        return "You can use this system to predict thyroid cancer recurrence or assess patient risk. Just fill in the clinical details in the form and click the predict button."
    
    if any(word in message for word in ['risk', 'assessment']):
        return "Risk assessment classifies patients into Low, High, or Intermediate categories based on clinical features like T, N, M staging and age."
    
    if any(word in message for word in ['recurrence', 'predict recurrence']):
        return "Recurrence prediction determines if the cancer is likely to return ('Yes' or 'No') based on clinical parameters and patient response."
    
    if any(word in message for word in ['cancer', 'thyroid']):
        return "Thyroid cancer is a disease where malignant cells form in the tissues of the thyroid gland. Early detection and risk assessment are crucial for effective treatment."
    
    if any(word in message for word in ['model', 'accuracy', 'algorithm']):
        return "Our system uses advanced XGBoost and Logistic Regression models trained on clinical datasets to provide highly accurate predictions."

    if any(word in message for word in ['who', 'creator', 'project']):
        return "This project was developed to assist medical professionals in evaluating thyroid cancer outcomes using Machine Learning."

    return "I'm not sure I understand. Could you please rephrase? You can ask me about risk assessment, recurrence prediction, or how to use the system."

if __name__ == '__main__':
    app.run(debug=True)

