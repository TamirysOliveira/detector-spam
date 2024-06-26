import streamlit as st
import numpy as np

# Dados de treinamento: [E1, E2, E3]
X = np.array([
    [5, 1, 0],  # Email 1
    [1, 0, 1],  # Email 2
    [3, 1, 0],  # Email 3
    [4, 1, 1],  # Email 4
    [0, 0, 0],  # Email 5
    [2, 0, 1],  # Email 6
])

# Rótulos (0 = Não Spam, 1 = Spam)
y = np.array([1, 0, 1, 1, 0, 0])

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000, weights=None, bias=None):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = weights
        self.bias = bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_features)
        if self.bias is None:
            self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    @staticmethod
    def _unit_step_func(x):
        return np.where(x >= 0, 1, 0)

# Interface do Streamlit
st.title("Detecção de Spam em Emails com Perceptron")

st.write("Insira as características do email para prever se é spam ou não:")

e1 = st.number_input("Número de palavras específicas (ex: 'grátis', 'promoção')", min_value=0, step=1)
e2 = st.selectbox("Presença de links", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
e3 = st.selectbox("Remetente não confiável", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

st.write("Ajuste os pesos e o bias do perceptron:")

w1 = st.number_input("Peso para número de palavras específicas (w1)", value=2.0)
w2 = st.number_input("Peso para presença de links (w2)", value=1.0)
w3 = st.number_input("Peso para remetente não confiável (w3)", value=1.5)
b = st.number_input("Bias", value=-2.0)

# Inicializar o perceptron com os pesos e bias fornecidos pelo usuário
perceptron = Perceptron(learning_rate=0.1, n_iters=10, weights=np.array([w1, w2, w3]), bias=b)
perceptron.fit(X, y)

if st.button("Verificar"):
    X_novo = np.array([[e1, e2, e3]])
    predicao = perceptron.predict(X_novo)
    
    if predicao[0] == 1:
        st.write("O email é considerado **spam**.")
    else:
        st.write("O email **não** é considerado spam.")
