from setuptools import setup, find_packages

setup(
    name="psychdr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "transformers>=4.20.0",
        "openpyxl",
        "lxml",
        "tqdm"
    ],
)