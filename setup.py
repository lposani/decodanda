from setuptools import setup, find_packages

VERSION = '0.5.0'
DESCRIPTION = 'A python package for neural decoding with built-in best practices.'
LONG_DESCRIPTION = 'Decodanda (dog latin for "to be decoded") is a best-practices-made-easy Python package for decoding neural data. Decodanda is designed to expose a user-friendly and flexible interface for population activity decoding, with a series of built-in best practices to avoid the most common pitfalls. In addition, Decodanda exposes a series of functions to compute the Cross-Condition Generalization Performance (CCGP, Bernardi et al. 2020) for the geometrical analysis of neural population activity.'

setup(
    name="decodanda",
    version=VERSION,
    author="Lorenzo Posani",
    author_email="lorenzo.posani@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.3.4', 'numpy>=1.20.1', 'pandas>=1.4.0', 'scikit_learn>=0.24.1',
        'scipy>=1.6.1', 'seaborn>=0.11.1', 'tqdm>=4.58.0'],
    keywords=['python', 'decoding', 'neuroscience', 'ccgp', 'neural activity', 'population activity', 'neural decoding', 'geometry'],
    url="https://github.com/lposani/decodanda",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
