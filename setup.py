from setuptools import setup, find_packages

setup(
    name="ATF",
    version="0.1.0",
    packages=find_packages(),
    description="A dynamic and adaptive function library for dataflow control.",
    author="",
    author_email="",
    url="",
    install_requires=[
        "tensorflow",
        "numpy",
        "networkx",
        "pandas",
        "matplotlib",
        "graphviz",
        "scipy",
        "pytest",
        "pygraphviz",
        "psutil",
        "tqdm",
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)