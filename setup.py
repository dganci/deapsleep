from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [r.strip() for r in f if r.strip()]

setup(
    name="deapsleep",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "deap==1.4.3",
        "matplotlib==3.10.3",
        "numpy==2.3.0",
        "optuna==4.3.0",
        "pandas==2.3.0",
        "pymoo==0.6.1.3",
        "PyYAML==6.0.2",
        "PyYAML==6.0.2",
        "scipy==1.15.3",
        "seaborn==0.13.2",
        "setuptools==72.1.0",
        "tabulate==0.9.0",
        "tqdm==4.67.1"
    ],
    include_package_data=True,
    license="GNU Lesser General Public License v3 (LGPLv3)",
    description="DeapSleep is a DEAP-based evolutionary computation toolkit for testing dropout in genetic algorithms",
    #long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="D. Ganci",
    author_email="daniele.ganci@studio.unibo.it",
    url="https://github.com/dganci/deapsleep",
    entry_points={
        "console_scripts": [
            "deapsleep = deapsleep.start:main",
        ],
    },
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Operating System :: OS Independent",
    ]
)