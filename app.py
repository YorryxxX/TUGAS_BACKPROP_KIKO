from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model & scaler
model = joblib.load('model.pkl')
scaler_X = joblib.load('scalerX.pkl')
scaler_y = joblib.load('scalerY.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    data = np.array([data])

    data_scaled = scaler_X.transform(data)
    pred = model.predict(data_scaled)
    hasil = scaler_y.inverse_transform(pred)

    return render_template('index.html', prediction=round(hasil[0][0],2))

if __name__ == "__main__":
    app.run(debug=True)