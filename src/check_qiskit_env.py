import sys

print("=" * 60)
print("CHECKING QUANTUM COMPUTING ENVIRONMENT")
print("=" * 60)
print(f"Python Version: {sys.version}")

try:
    import qiskit
    print(f"Qiskit Version: {qiskit.__version__}")
except ImportError:
    print("Qiskit Version: NOT INSTALLED")

try:
    import qiskit_machine_learning
    print(f"Qiskit Machine Learning Version: {qiskit_machine_learning.__version__}")
except ImportError:
    print("Qiskit Machine Learning Version: NOT INSTALLED")

try:
    import qiskit_algorithms
    print(f"Qiskit Algorithms Version: {qiskit_algorithms.__version__}")
except ImportError:
    print("Qiskit Algorithms Version: NOT INSTALLED")

print("=" * 60)
