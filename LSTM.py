from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Gerando dados sintéticos
data = np.random.randn(1000, 50)

# Criando a rede neural
model = Sequential()
model.add(LSTM(32, input_shape=(1, 50)))
model.add(Dense(1, activation='sigmoid'))

# Compilando a rede neural
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dividindo os dados em treinamento e teste
train_data = data[:800]
test_data = data[800:]

# Preparando os dados de entrada e saída
def prepare_data(data):
    X = data[:, :-1]
    y = data[:, -1]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# Treinando a rede neural
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Avaliando a rede neural
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
