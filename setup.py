from setuptools import setup, find_packages

VERSION = '0.0.2'
DESCRIPTION = 'Best practices made easy for decoding in neuroscience.'
LONG_DESCRIPTION = 'Decodanda is a best-practices-made-easy Python package that handles the most common issues in decoding from population activity in neuroscience.'
# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
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
    keywords=['python', 'decoding', 'neuroscience', 'ccgp', 'neural activity', 'population activity', 'neural decoding'],
    url="https://github.com/lposani/decodanda",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)