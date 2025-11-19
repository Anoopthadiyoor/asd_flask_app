# app.py
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for

# Load preprorocesor
try:
    lda = joblib.load('lda_asd_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    print("Model and preprocessor loaded successfully.")
except FileNotFoundError:
    print("Error: Model or preprocessor files not found.")
    lda = None
    preprocessor = None

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/assessment')
def assessment():
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        feature_names = [f for f in preprocessor.feature_names_in_ if f != 'Class']
    else:
        feature_names = []
    return render_template('assessment.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if lda is None or preprocessor is None:
        return jsonify({'error': 'Model or preprocessor not loaded.'}), 500

    try:
        form_data = request.form.to_dict()
        input_data = pd.DataFrame([form_data])

        # Trim strings and normalize
        input_data = input_data.applymap(lambda v: v.strip() if isinstance(v, str) else v)

        expected_features = [f for f in preprocessor.feature_names_in_ if f != 'Class']

        missing = [f for f in expected_features if f not in input_data.columns]
        if missing:
            raise ValueError(f"Missing input fields: {missing}.")

        # Convert input to numeric (tolerant mapping)
        for col in input_data.columns:
            if col == 'Age':
                input_data[col] = input_data[col].replace('', np.nan).astype(float)
            else:
                try:
                    input_data[col] = input_data[col].astype(int)
                except Exception:
                    mapping = {
                        'agree': 1, 'disagree': 0,
                        'yes': 1, 'no': 0,
                        'male': 1, 'female': 0,
                        'm': 1, 'f': 0,
                        '1': 1, '0': 0
                    }
                    input_data[col] = input_data[col].astype(str).str.strip().str.lower().map(mapping)
                    if input_data[col].isnull().any():
                        bad_vals = input_data[col].loc[input_data[col].isnull()].tolist()
                        raise ValueError(f"Unrecognized value(s) for '{col}': {bad_vals}.")
                    input_data[col] = input_data[col].astype(int)

        input_data = input_data[expected_features]

        input_preprocessed = preprocessor.transform(input_data)
        prediction = lda.predict(input_preprocessed)
        probability = lda.predict_proba(input_preprocessed)[:, 1]

        if probability[0] > 0.7:
            chance_label = "High likelihood of ASD"
            color = "red"
        elif probability[0] > 0.4:
            chance_label = "Moderate likelihood of ASD"
            color = "orange"
        else:
            chance_label = "Low likelihood of ASD"
            color = "green"

        return redirect(url_for('results',
                                prediction=int(prediction[0]),
                                chance=chance_label,
                                color=color))
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)}), 400

@app.route('/results')
def results():
    prediction = request.args.get('prediction')
    chance = request.args.get('chance')
    color = request.args.get('color', 'gray')

    if prediction is None or chance is None:
        return "Error: Results not found.", 400

    return render_template('results.html',
                           prediction=prediction,
                           chance=chance,
                           color=color)

if __name__ == '__main__':
    app.run(debug=True)
