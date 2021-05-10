import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

from debiasmedimg import (
    __name__, __version__,
    __author__,
    __author_email__,
    __description__,
    __url__
)

setuptools.setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
