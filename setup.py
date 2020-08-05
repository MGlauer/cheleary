from setuptools import setup
from setuptools import find_packages
from glob import glob
from os.path import basename
from os.path import splitext

setup(
    name="cheleary",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/MGlauer/cheleary",
    license="",
    install_requires=["tensorflow", "tensorflow_addons", "numpy", "pysmiles", "pandas"],
    extras_require={"dev": ["black", "pre-commit"]},
    author="glauer",
    author_email="",
    description="Cheleary is a toolkit to build an easy training environment. It implements \
                 different kinds of encodings and network structures based on `keras` and \
                 `tensorflow`. The main focus are learning tasks around `CHEBI` - an \
                 ontology about chemicals",
    entry_points={"console_scripts": ["cheleary = cheleary.cli:cli"]},
)
