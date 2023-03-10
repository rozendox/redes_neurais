from keras.models import Sequential
from keras.layers import GRU, Dense
import numpy as np

# Dados de treinamento
X_train = np.array([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
                    [[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
                    [[0.3, 0.4], [0.4, 0.5], [0.5, 0.6]],
                    [[0.4, 0.5], [0.5, 0.6], [0.6, 0.7]]])
y_train = np.array([[0.4], [0.5], [0.6], [0.7]])

# Criação do modelo
model = Sequential()
model.add(GRU(4, input_shape=(3, 2)))
model.add(Dense(1))

# Compilação do modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Treinamento do modelo
model.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=2)

# Predição com dados de teste
X_test = np.array([[[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]],
                   [[0.6, 0.7], [0.7, 0.8], [0.8, 0.9]]])
y_test = np.array([[0.8], [0.9]])

y_pred = model.predict(X_test)

print('y_test:\n', y_test)
print('y_pred:\n', y_pred)
