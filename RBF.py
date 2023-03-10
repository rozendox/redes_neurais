from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Gerando dados sintéticos
X, y = make_blobs(n_samples=500, centers=3, random_state=42)

# Criando o modelo de clusterização
kmeans = KMeans(n_clusters=3, random_state=42)

# Ajustando o modelo de clusterização
kmeans.fit(X)

# Obtendo as distâncias dos centros para cada ponto
distances = kmeans.transform(X)

# Criando a matriz de entrada da RBN
X_rbn = np.exp(-(distances ** 2) / (2 * (np.mean(distances) ** 2)))

# Normalizando a matriz de entrada
scaler = StandardScaler()
X_rbn = scaler.fit_transform(X_rbn)

# Criando o modelo da RBN
rbn = MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', solver='lbfgs')

# Treinando o modelo da RBN
rbn.fit(X_rbn, y)

# Avaliando a acurácia do modelo
y_pred = rbn.predict(X_rbn)
accuracy = accuracy_score(y, y_pred)
print('Accuracy:', accuracy)
