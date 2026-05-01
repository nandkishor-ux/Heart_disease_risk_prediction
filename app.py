from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURE_COLS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Heart Disease Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: Arial, sans-serif; background: #f0f2f5; display: flex; justify-content: center; padding: 30px 10px; }
        .card { background: white; border-radius: 12px; padding: 30px; max-width: 750px; width: 100%; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }
        h1 { color: #c0392b; margin-bottom: 6px; font-size: 24px; }
        p.sub { color: #666; margin-bottom: 24px; font-size: 14px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
        label { font-size: 13px; color: #444; display: block; margin-bottom: 4px; font-weight: bold; }
        input, select { width: 100%; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #c0392b; }
        hr { border: none; border-top: 1px solid #eee; margin: 24px 0; }
        button { width: 100%; padding: 12px; background: #c0392b; color: white; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; }
        button:hover { background: #a93226; }
        .result { margin-top: 24px; padding: 16px; border-radius: 8px; display: none; }
        .result.show { display: block; }
        .result.low { background: #eafaf1; border: 1px solid #27ae60; }
        .result.moderate { background: #fef9e7; border: 1px solid #f39c12; }
        .result.high { background: #fdedec; border: 1px solid #c0392b; }
        .result h2 { font-size: 18px; margin-bottom: 8px; }
        .metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 12px; }
        .metric { background: rgba(0,0,0,0.04); border-radius: 6px; padding: 10px; text-align: center; }
        .metric span { display: block; font-size: 12px; color: #666; }
        .metric strong { font-size: 18px; }
    </style>
</head>
<body>
<div class="card">
    <h1>🫀 Heart Disease Risk Predictor</h1>
    <p class="sub">Fill in patient details and click Predict.</p>
    <div class="grid">
        <div><label>Age</label><input type="number" id="age" value="50" min="1" max="120"></div>
        <div><label>Sex</label><select id="sex"><option value="0">Female</option><option value="1" selected>Male</option></select></div>
        <div><label>Chest Pain Type</label><select id="cp"><option value="0">Typical Angina</option><option value="1">Atypical Angina</option><option value="2">Non-anginal Pain</option><option value="3">Asymptomatic</option></select></div>
        <div><label>Resting BP (mmHg)</label><input type="number" id="trestbps" value="120" min="50" max="250"></div>
        <div><label>Cholesterol (mg/dl)</label><input type="number" id="chol" value="200" min="100" max="600"></div>
        <div><label>Fasting Blood Sugar &gt;120?</label><select id="fbs"><option value="0">No</option><option value="1">Yes</option></select></div>
        <div><label>Resting ECG</label><select id="restecg"><option value="0">Normal</option><option value="1">ST-T Abnormality</option><option value="2">LV Hypertrophy</option></select></div>
        <div><label>Max Heart Rate</label><input type="number" id="thalach" value="150" min="60" max="250"></div>
        <div><label>Exercise Induced Angina</label><select id="exang"><option value="0">No</option><option value="1">Yes</option></select></div>
        <div><label>ST Depression (oldpeak)</label><input type="number" id="oldpeak" value="1.0" step="0.1" min="0" max="10"></div>
        <div><label>Slope of ST Segment</label><select id="slope"><option value="0">Upsloping</option><option value="1">Flat</option><option value="2">Downsloping</option></select></div>
        <div><label>Major Vessels (0-3)</label><select id="ca"><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></div>
        <div><label>Thalassemia</label><select id="thal"><option value="0">Unknown</option><option value="1">Normal</option><option value="2">Fixed Defect</option><option value="3">Reversible Defect</option></select></div>
    </div>
    <hr>
    <button onclick="predict()">🔍 Predict</button>
    <div class="result" id="result">
        <h2 id="result-title"></h2>
        <div class="metrics">
            <div class="metric"><span>Prediction</span><strong id="m-pred"></strong></div>
            <div class="metric"><span>Probability</span><strong id="m-prob"></strong></div>
            <div class="metric"><span>Risk Level</span><strong id="m-risk"></strong></div>
        </div>
        <p style="font-size:12px;color:#999;margin-top:12px;">⚠️ For educational purposes only. Not a medical diagnosis.</p>
    </div>
</div>
<script>
async function predict() {
    const data = {
        age: +document.getElementById('age').value,
        sex: +document.getElementById('sex').value,
        cp: +document.getElementById('cp').value,
        trestbps: +document.getElementById('trestbps').value,
        chol: +document.getElementById('chol').value,
        fbs: +document.getElementById('fbs').value,
        restecg: +document.getElementById('restecg').value,
        thalach: +document.getElementById('thalach').value,
        exang: +document.getElementById('exang').value,
        oldpeak: +document.getElementById('oldpeak').value,
        slope: +document.getElementById('slope').value,
        ca: +document.getElementById('ca').value,
        thal: +document.getElementById('thal').value
    };
    const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
    const r = await res.json();
    const box = document.getElementById('result');
    box.className = 'result show ' + (r.risk_level.includes('LOW') ? 'low' : r.risk_level.includes('MODERATE') ? 'moderate' : 'high');
    document.getElementById('result-title').innerText = r.message;
    document.getElementById('m-pred').innerText = r.prediction == 1 ? 'Disease' : 'No Disease';
    document.getElementById('m-prob').innerText = (r.probability * 100).toFixed(1) + '%';
    document.getElementById('m-risk').innerText = r.risk_level;
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])[FEATURE_COLS]
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]
    probability = float(model.predict_proba(scaled)[0][1])
    if probability < 0.3:
        risk = "LOW RISK"
    elif probability < 0.6:
        risk = "MODERATE RISK"
    else:
        risk = "HIGH RISK"
    return jsonify({
        'prediction': int(prediction),
        'probability': round(probability, 4),
        'risk_level': risk,
        'message': 'Heart disease detected.' if prediction == 1 else 'No heart disease detected.'
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, port=5000)