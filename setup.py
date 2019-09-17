import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
        return f.read()


setup(
    name='SchNOrb',
    version='0.1',
    author="Kristof T. Schuett, Michael Gastegger, Reinhard Maurer",
    url="https://github.com/atomistic-machine-learning/SchNOrb",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.1",
        "numpy",
        "schnetpack>=0.3",
        "spherical_functions",
        "tqdm",
    ],
    scripts=[
        "src/scripts/run_schnorb.py",
        "src/scripts/extract_data.py",
    ],
    description='Unifying machine learning and quantum chemistry with a deep neural network for molecular wavefunctions',
    long_description=read('README.md')
)
