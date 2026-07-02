# setup.py
from setuptools import setup, find_packages

setup(
    name="surrogate-fem",
    version="0.2.0",
    packages=find_packages(),
    py_modules=["material_params"],
)