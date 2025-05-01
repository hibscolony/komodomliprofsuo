from setuptools import setup, find_packages

setup(
    name="komodomliprofsuo",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
    author="Koding Muda Nusantara",
    description="Komodo Mlipir Algorithm for Hyperparameter Optimization",
    url="https://github.com/hibscolony/komodomliprofsuo",
)
