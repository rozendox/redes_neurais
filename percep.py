import numpy as np

# Definindo a função de ativação degrau (step function)
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# Definindo a classe Perceptron
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0
        
    def predict(self, inputs):
        # Multiplicando os pesos pelas entradas e somando o bias
        z = np.dot(inputs, self.weights) + self.bias
        # Aplicando a função de ativação
        a = step_function(z)
        return a
        
    def train(self, inputs, labels, learning_rate):
        # Iterando pelo número de épocas
        for _ in range(100):
            # Iterando pelas entradas e etiquetas
            for i in range(len(inputs)):
                x = inputs[i]
                y = labels[i]
                # Calculando a saída da rede
                output = self.predict(x)
                # Atualizando os pesos e o bias
                self.weights += learning_rate * (y - output) * x
                self.bias += learning_rate * (y - output)
