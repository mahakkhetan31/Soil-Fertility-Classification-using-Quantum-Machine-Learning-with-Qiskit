import qiskit
from qiskit.circuit.library import ZZFeatureMap, zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("Testing Feature Map Instantiation...")
fm1 = zz_feature_map(feature_dimension=4, reps=2, entanglement='linear')
print(f"zz_feature_map created successfully! Qubits: {fm1.num_qubits}")

qk = FidelityQuantumKernel(feature_map=fm1)
print(f"FidelityQuantumKernel instantiated successfully!")

import numpy as np
sample_x = np.random.uniform(0, np.pi, (5, 4))
res = qk.evaluate(sample_x, sample_x)
print(f"Kernel Matrix Evaluated! Shape: {res.shape}")
print("Succeeded!")
