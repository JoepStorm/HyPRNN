# Editable-install packaging: puts the scripts_* packages and the top-level
# material_params module on the import path (`pip install -e .`).
from setuptools import setup, find_packages

setup(
    name="surrogate-fem",
    version="0.2.0",
    packages=find_packages(),
    py_modules=["material_params"],
)