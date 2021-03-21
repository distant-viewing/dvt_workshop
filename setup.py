from setuptools import setup
from setuptools import find_packages

long_description= """
The Distant TV Toolkit is a Python package designed to facilitate the
computational analysis of visual culture. This is a special module containing
experimental code used in our workshops.
"""

required = [
    "detectron2",
    "keras",
    "numpy"
]

setup(
    name="dvt_workshop",
    version="0.0.1",
    description="Analysis of Visual Culture at Scale",
    long_description=long_description,
    author="Taylor Anold, Lauren Tilton",
    author_email="taylor.arnold@acm.org",
    url="https://github.com/distant-viewing/dvt_workshop",
    license="GPL-2",
    install_requires=required,
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or "
        "later (LGPLv2+)",
        "Programming Language :: Python :: 3.7",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(),
)
