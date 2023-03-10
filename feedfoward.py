from keras.models import Sequential
from keras.layers import Dense

# Criando a rede neural
model = Sequential()
model.add(Dense(64, input_dim=100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilando a rede neural
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando a rede neural
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Avaliando a rede neural
loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)
