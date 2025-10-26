from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [r.strip() for r in f if r.strip()]

setup(
    name="deapsleep",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "tk", "pandas", "numpy", "deap", "pymoo"
    ],
    entry_points={
        "console_scripts": [
            "deapsleep = deapsleep.start:main",
        ],
    },
)