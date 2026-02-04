from setuptools import setup, find_packages

setup(
    name="aog-vascular",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "trimesh>=4.0.0",
    ],
    python_requires=">=3.10",
)
