import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# =====================
# LOAD DATA
# =====================
df = pd.read_csv('student_data.csv')

# Ambil kolom penting
df = df[['studytime','failures','absences','G1','G2','G3']]

# =====================
# SPLIT X & y
# =====================
X = df[['studytime','failures','absences','G1','G2']].values
y = df['G3'].values.reshape(-1,1)

# =====================
# NORMALISASI
# =====================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# =====================
# TRAIN TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# MODEL BACKPROPAGATION
# =====================
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=10)

# =====================
# TRAINING
# =====================
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# =====================
# EVALUASI
# =====================
loss = model.evaluate(X_test, y_test)
print("Loss:", loss)

# =====================
# SAVE MODEL
# =====================
joblib.dump(model, 'model.pkl')
joblib.dump(scaler_X, 'scalerX.pkl')
joblib.dump(scaler_y, 'scalerY.pkl')

# =====================
# PREDIKSI CONTOH
# =====================
data_baru = np.array([[2, 0, 4, 10, 12]])

data_scaled = scaler_X.transform(data_baru)
pred = model.predict(data_scaled)

hasil = scaler_y.inverse_transform(pred)

print("Prediksi nilai akhir:", hasil[0][0])