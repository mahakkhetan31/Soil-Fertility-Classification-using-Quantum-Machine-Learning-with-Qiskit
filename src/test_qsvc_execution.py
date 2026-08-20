import qiskit
import qiskit_machine_learning
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import zz_feature_map
import numpy as np

print("Testing QSVC Instantiation & Execution...")
feature_map = zz_feature_map(feature_dimension=4, reps=2, entanglement='linear')
q_kernel = FidelityQuantumKernel(feature_map=feature_map)

qsvc = QSVC(quantum_kernel=q_kernel)
print("QSVC successfully instantiated!")

X_tr = np.random.uniform(0, np.pi, (20, 4))
y_tr = np.random.choice([0, 1, 2], size=20)
qsvc.fit(X_tr, y_tr)
print("QSVC fit succeeded!")

preds = qsvc.predict(X_tr[:5])
print(f"Predictions: {preds}")
