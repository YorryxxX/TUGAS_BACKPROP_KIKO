import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib

# load data
df = pd.read_csv('student_data.csv')

# ambil fitur
X = df[['studytime','failures','absences','G1','G2']]
y = df[['G3']]

# scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)

# model backpropagation
model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=500)

model.fit(X_train, y_train.ravel())

# save
joblib.dump(model, 'model.pkl')
joblib.dump(scaler_X, 'scalerX.pkl')
joblib.dump(scaler_y, 'scalerY.pkl')

print("Model berhasil disimpan!")