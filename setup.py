from setuptools import setup

setup(
    name="cheleary",
    version="",
    packages=["src", "src.cheleary"],
    url="https://github.com/MGlauer/cheleary",
    license="",
    install_requires=["tensorflow", "numpy", "pysmiles", "pandas"],
    extras_require={
        'dev': [
            'black',
            'pre-commit'
        ]
    },
    author="glauer",
    author_email="",
    description="",
)
