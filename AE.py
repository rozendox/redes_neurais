from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# Dados de treinamento
X_train = np.random.rand(1000, 100)

# Criação do modelo
input_layer = Input(shape=(100,))
encoded = Dense(50, activation='relu')(input_layer)
decoded = Dense(100, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)

# Compilação do modelo
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Treinamento do modelo
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Predição com dados de teste
X_test = np.random.rand(10, 100)
decoded_output = autoencoder.predict(X_test)

print('Entrada original:\n', X_test)
print('Saída reconstruída:\n', decoded_output)
