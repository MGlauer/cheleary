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
    description="",
)
