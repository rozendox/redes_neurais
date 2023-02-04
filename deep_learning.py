import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(X_treinamento[0], cmap = 'gray')
plt.title(y_treinamento[0])

plt.subplot(2, 2, 2)
plt.imshow(X_treinamento[1], cmap = 'gray')
plt.title(y_treinamento[1])

plt.subplot(2, 2, 3)
plt.imshow(X_treinamento[3], cmap = 'gray')
plt.title(y_treinamento[3])

plt.subplot(2, 2, 4)
plt.imshow(X_treinamento[4], cmap = 'gray')
plt.title(y_treinamento[4])

X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))

X_treinamento = X_treinamento.astype('float32')
X_teste = X_teste.astype('float32')

X_treinamento /= 255
X_teste /= 255

y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

# 784 - 64 - 64 - 64 - 10
modelo = Sequential()
modelo.add(Dense(units = 64, activation = 'relu', input_dim = 784))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 64, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 64, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(units = 10, activation = 'softmax'))

modelo.summary()

modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
historico = modelo.fit(X_treinamento, y_treinamento, epochs = 20,
                       validation_data = (X_teste, y_teste))

historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_acc'])

previsoes = modelo.predict(X_teste)
y_teste_matriz = [np.argmax(t) for t in y_teste]
y_previsoes_matriz = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_teste_matriz, y_previsoes_matriz)

y_treinamento[20]
novo = X_treinamento[20]
novo = np.expand_dims(novo, axis = 0)
pred = modelo.predict(novo)


















