from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler_X = joblib.load(os.path.join(BASE_DIR, 'scalerX.pkl'))
scaler_y = joblib.load(os.path.join(BASE_DIR, 'scalerY.pkl'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        data = np.array([data])

        data_scaled = scaler_X.transform(data)
        pred = model.predict(data_scaled)

        hasil = scaler_y.inverse_transform(pred.reshape(-1,1))

        return render_template('index.html', prediction=round(hasil[0][0],2))
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)