# Definindo a classe Perceptron
Perceptron <- function(num_inputs) {
  weights <- rnorm(num_inputs)
  bias <- rnorm(1)
  # Função de ativação step
  activate <- function(x) {
    if (x > 0) {
      return(1)
    } else {
      return(0)
    }
  }
  # Função para fazer a predição
  predict <- function(inputs) {
    weighted_sum <- bias
    for (i in 1:length(inputs)) {
      weighted_sum <- weighted_sum + weights[i] * inputs[i]
    }
    return(activate(weighted_sum))
  }
  # Retornando as funções como um objeto
  list(predict = predict)
}

# Criando um Perceptron com 2 entradas
p <- Perceptron(2)

# Dados de treinamento
training_data <- data.frame(inputs = list(c(0, 0), c(0, 1), c(1, 0), c(1, 1)), label = c(0, 0, 0, 1))

# Treinando o Perceptron
learning_rate <- 0.1
num_epochs <- 10
for (epoch in 1:num_epochs) {
  for (i in 1:nrow(training_data)) {
    inputs <- training_data$inputs[[i]]
    label <- training_data$label[i]
    prediction <- p$predict(inputs)
    error <- label - prediction
    p$bias <- p$bias + learning_rate * error
    for (j in 1:length(inputs)) {
      p$weights[j] <- p$weights[j] + learning_rate * error * inputs[j]
    }
  }
}

# Fazendo predições com o Perceptron treinado
cat("0 0: ", p$predict(c(0, 0)), "\n")
cat("0 1: ", p$predict(c(0, 1)), "\n")
cat("1 0: ", p$predict(c(1, 0)), "\n")
cat("1 1: ", p$predict(c(1, 1)), "\n")
