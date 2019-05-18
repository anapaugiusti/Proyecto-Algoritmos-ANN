from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

w1 = np.array([[1.38046736, 0.59503797, 0.31349674, 0.74832673],
        [0.98004188, 0.62400932, 0.95052872, 0.63898124]])

w2 = np.array([[1.57303161, 1.74623166],
        [1.53422492, 1.56381663]])

w3 = np.array([[11.07510303, 10.46564949]])

b1 = np.array([0.81943422, 0.49194958])
b2 = np.array([0.37197514, 0.76565576])
b3 = np.array([11.07261949])

data = pd.read_csv("Reporte_cuervo.csv")

X1 = np.array([[float(i) for i in data['CIERRE'][::-1][:-1]]]).T
X2 = np.array([[float(i) for i in data['CIERRE'][::-1][:-2]]]).T
X3 = np.array([[float(i) for i in data['CIERRE'][::-1][:-3]]]).T
X4 = np.array([[float(i) for i in data['CIERRE'][::-1][:-4]]]).T
X = np.append(X1[1:], X2, axis=1)
X = np.append(X[1:], X3, axis=1)
X = np.append(X[1:], X4, axis=1)

y = np.array([data['CIERRE'][::-1][4:]]).T

X_scale = StandardScaler()

test = -1

X = X_scale.fit_transform(X)

W = [w1, w2, w3]
b = [b1, b2, b3]

def f(x):
    return(1 / (1 + np.exp(-x)))

def model(n_layers, x, w, b):
    for l in range(n_layers - 1):
        if l == 0:
            node_in = x
        else:
            node_in = h.T
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return z

print(f"Estimado hoy: {model(4, X[test], W, b)}\nReal: {y[test]}")
#print(f"Estimado dia siguiente: {model(4, X[test+1], W, b)}\nReal: {y[test+1]}")
result = model(4, X[test], W, b)
#%% Para día siguiente tomando último dato disponible
X1 = np.array([[float(i) for i in data['CIERRE'][::-1][:-1]]]).T
X2 = np.array([[float(i) for i in data['CIERRE'][::-1][:-2]]]).T
X3 = np.array([[float(i) for i in data['CIERRE'][::-1][:-3]]]).T
X4 = np.array([[float(i) for i in data['CIERRE'][::-1][:-4]]]).T

X4 = np.append(X4, X3[-1])
X3 = np.append(X3, X2[-1])
X2 = np.append(X2, X1[-1])
X1 = np.append(X1, result[0])

X1 = X1.reshape(len(X1), 1)
X2 = X2.reshape(len(X2), 1)
X3 = X3.reshape(len(X3), 1)
X4 = X4.reshape(len(X4), 1)

Xt = np.append(X1[1:], X2, axis=1)
Xt = np.append(Xt[1:], X3, axis=1)
Xt = np.append(Xt[1:], X4, axis=1)

print(f"Estimado mañana: {model(4, Xt[-1], W, b)}")
