from setuptools import setup, find_packages

setup(
    name="ClimbInterp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "cvxpy"
    ],
    author="SAFE SOLUTIONS",
    description="An interpolation method that aims to maintain conservatism across a set of data. The strategy used is to assume exponential behaviour from a certain point, however until the interpolation gets to this point, other forms of interpolation are used and interconnected, creating two functions that complement each other.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
