from setuptools import setup
from setuptools import find_packages

setup(
    name="cheleary",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/MGlauer/cheleary",
    license="",
    install_requires=["tensorflow", "numpy", "pysmiles", "pandas"],
    extras_require={"dev": ["black", "pre-commit"]},
    author="glauer",
    author_email="",
    description="Cheleary is a toolkit to build an easy training environment. It implements \
                 different kinds of encodings and network structures based on `keras` and \
                 `tensorflow`. The main focus are learning tasks around `CHEBI` - an \
                 ontology about chemicals",
)
