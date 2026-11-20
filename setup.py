from setuptools import setup, find_packages

setup(
    name="psychdr",
    version="0.1.0",
    description="Knowledge graph-driven drug repurposing framework for psychiatric disorders",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "transformers==4.53.3",
        "huggingface-hub==0.34.0",
        "numpy==1.26.4",
        "pandas==1.5.3",
        "scikit-learn==1.7.0",
        "scipy",
        "openpyxl==3.1.5",
        "lxml",
        "tqdm",
    ],
)
