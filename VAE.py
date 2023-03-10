from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import mse
import keras.backend as K
import numpy as np

# Dados de treinamento
X_train = np.random.rand(1000, 100)

# Tamanho da representação latente
latent_dim = 10

# Criação do encoder
input_layer = Input(shape=(100,))
hidden_layer = Dense(50, activation='relu')(input_layer)
z_mean = Dense(latent_dim)(hidden_layer)
z_log_var = Dense(latent_dim)(hidden_layer)

# Amostragem da distribuição normal gaussiana
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Criação do decoder
decoder_hidden_layer = Dense(50, activation='relu')
decoder_output_layer = Dense(100, activation='sigmoid')
hidden_decoder = decoder_hidden_layer(z)
output_decoder = decoder_output_layer(hidden_decoder)

# Criação do modelo completo
vae = Model(input_layer, output_decoder)

# Definição da função de perda
reconstruction_loss = mse(input_layer, output_decoder)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compilação do modelo
vae.compile(optimizer='adam')

# Treinamento do modelo
vae.fit(X_train, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Predição com dados de teste
X_test = np.random.rand(10, 100)
decoded_output = vae.predict(X_test)

print('Entrada original:\n', X_test)
print('Saída reconstruída:\n', decoded_output)
