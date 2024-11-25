from setuptools import setup, find_packages

setup(
    name='gap_tools_ed',
    version='1.0',
    description='A package to compute the gap and the annealing time for quantum annealing in the transverse field Ising model. The package is based on QuTiP and NumPy. It uses exact diagonilization of the Hamiltonian.',
    author='Romain Piron',
    packages=find_packages(),
)