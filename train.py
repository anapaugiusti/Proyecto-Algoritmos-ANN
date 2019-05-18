from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy.random as r
import pandas as pd
import numpy as np

file = open('act_weights.txt', 'a+')

data = pd.read_csv("Reporte_cuervo.csv")

#Convertimos los datos a matriz
X1 = np.array([[float(i) for i in data['CIERRE'][::-1][:-1]]]).T
X2 = np.array([[float(i) for i in data['CIERRE'][::-1][:-2]]]).T
X3 = np.array([[float(i) for i in data['CIERRE'][::-1][:-3]]]).T
X4 = np.array([[float(i) for i in data['CIERRE'][::-1][:-4]]]).T
X = np.append(X1[1:], X2, axis=1)
X = np.append(X[1:], X3, axis=1)
X = np.append(X[1:], X4, axis=1)

X_scale = StandardScaler()

X = X_scale.fit_transform(X)

y = np.array([data['CIERRE'][::-1][4:]]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

nn_structure = [4, 2, 2, 1]    #[entradas, nodos, salida]

def f(x):
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            node_in = x
        else:
            node_in = h[l]
        z[l+1] = W[l].dot(node_in) + b[l]   # z^(l+1) = W^(l)*h^(l) + b^(l)  
        h[l+1] = f(z[l+1])                  # h^(l) = f(z^(l))
    return h, z

def calculate_out_layer_delta(y, h_out, z_out):
    one_array = np.ones(np.shape(z_out))
    return -(y-z_out) * one_array

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iteraciones=100, alpha=0.05):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    while cnt < iteraciones:
        if cnt%10 == 0:
            print('Iteracion {} de {}'.format(cnt, iteraciones))   
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            h, z = feed_forward(X[i, :], W, b)
            for l in range(len(nn_structure), 0, -1):
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    tri_b[l] += delta[l+1]
                    
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_train)

plt.plot(avg_cost_func)
plt.ylabel('Average J')
plt.xlabel('Iteration number')
plt.show()

def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = []
    for i in range(m):
        h, z = feed_forward(X[i, :], W, b)
        y.append([z[3][0]])
    return y


y_pred = predict_y(W, b, X_test, 4)
file.write("Weights")

print(f"Explained Variance Score: {explained_variance_score(y_test, y_pred)}")

for i in W:
    file.write('W')
    for j in W[i]:
        file.write(str(j) + "\n")

file.write("Bias")
for i in b:
    file.write(str(b) + "\n")
file.close()
