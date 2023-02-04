from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

iris = datasets.load_iris()

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(iris.data, iris.target,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

modelo = MLPClassifier(verbose = True, hidden_layer_sizes=(5,4), max_iter = 10000)
modelo.fit(X_treinamento, y_treinamento)

previsoes = modelo.predict(X_teste)
accuracy_score(y_teste, previsoes)

confusao = ConfusionMatrix(modelo)
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.poof()
