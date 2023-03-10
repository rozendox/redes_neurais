#include <iostream>
#include <vector>
#include <random>

using namespace std;

// Definindo a classe Perceptron
class Perceptron {
private:
    vector<double> weights;
    double bias;
public:
    Perceptron(int num_inputs) {
        // Inicializando os pesos aleatoriamente
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < num_inputs; i++) {
            weights.push_back(distribution(gen));
        }
        // Inicializando o bias aleatoriamente
        bias = distribution(gen);
    }
    // Função de ativação step
    double activate(double x) {
        if (x > 0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
    // Função para fazer a predição
    double predict(vector<double> inputs) {
        double weighted_sum = bias;
        for (int i = 0; i < inputs.size(); i++) {
            weighted_sum += weights[i] * inputs[i];
        }
        return activate(weighted_sum);
    }
};

int main() {
    // Criando um Perceptron com 2 entradas
    Perceptron p(2);
    // Dados de treinamento
    vector<vector<double>> training_data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> labels = {0, 0, 0, 1};
    // Treinando o Perceptron
    double learning_rate = 0.1;
    int num_epochs = 10;
    for (int i = 0; i < num_epochs; i++) {
        for (int j = 0; j < training_data.size(); j++) {
            vector<double> inputs = training_data[j];
            double label = labels[j];
            double prediction = p.predict(inputs);
            double error = label - prediction;
            p.bias += learning_rate * error;
            for (int k = 0; k < inputs.size(); k++) {
                p.weights[k] += learning_rate * error * inputs[k];
            }
        }
    }
    // Fazendo predições com o Perceptron treinado
    cout << "0 0: " << p.predict({0, 0}) << endl;
    cout << "0 1: " << p.predict({0, 1}) << endl;
    cout << "1 0: " << p.predict({1, 0}) << endl;
    cout << "1 1: " << p.predict({1, 1}) << endl;
    return 0;
}
