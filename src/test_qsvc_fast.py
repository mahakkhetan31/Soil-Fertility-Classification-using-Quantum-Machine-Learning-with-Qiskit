import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("Testing Fast Statevector Quantum Kernel SVC...")

# Generate 4-qubit feature map
fm = zz_feature_map(feature_dimension=4, reps=2, entanglement='linear')

def compute_q_kernel_matrix(X1, X2):
    sv1 = [Statevector(fm.assign_parameters(dict(zip(fm.parameters, x)))) for x in X1]
    sv2 = [Statevector(fm.assign_parameters(dict(zip(fm.parameters, x)))) for x in X2]
    K = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = np.abs(sv1[i].inner(sv2[j])) ** 2
    return K

X_tr = np.random.uniform(0, np.pi, (30, 4))
X_te = np.random.uniform(0, np.pi, (10, 4))
y_tr = np.random.choice([0, 1, 2], size=30)
y_te = np.random.choice([0, 1, 2], size=10)

K_train = compute_q_kernel_matrix(X_tr, X_tr)
K_test = compute_q_kernel_matrix(X_te, X_tr)

qsvc = SVC(kernel='precomputed')
qsvc.fit(K_train, y_tr)
preds = qsvc.predict(K_test)

print(f"Predictions: {preds}")
print(f"Test Accuracy: {accuracy_score(y_te, preds)*100:.2f}%")
print("Fast Statevector QSVC Succeeded!")
