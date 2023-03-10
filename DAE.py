from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

# Dados de treinamento
X_train = np.random.rand(1000, 100)

# Adição de ruído aos dados de entrada
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

# Criação do encoder
input_layer = Input(shape=(100,))
hidden_layer = Dense(50, activation='relu')(input_layer)
latent_layer = Dense(10, activation='relu')(hidden_layer)

# Criação do decoder
decoder_hidden_layer = Dense(50, activation='relu')
output_layer = Dense(100, activation='sigmoid')
hidden_decoder = decoder_hidden_layer(latent_layer)
output_decoder = output_layer(hidden_decoder)

# Criação do modelo completo
dae = Model(input_layer, output_decoder)

# Compilação do modelo
dae.compile(optimizer='adam', loss='mse')

# Treinamento do modelo
dae.fit(X_train_noisy, X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Predição com dados de teste
X_test = np.random.rand(10, 100)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
decoded_output = dae.predict(X_test_noisy)

print('Entrada original:\n', X_test)
print('Saída reconstruída:\n', decoded_output)
