from setuptools import find_packages, setup

setup(
    name="src",
    version="0.1.0",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.10",
)
