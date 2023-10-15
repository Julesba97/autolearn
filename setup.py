from setuptools import find_packages, setup
from typing import List
from pathlib import Path

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    return requirements

with open("./README.md", "r") as f:
    long_description = f.read()

__version__ = "0.0.0"
requirements_path = Path("./requirements.txt")
setup(
    name="autolearn",
    version=__version__,
    description="Le package autolearn est conçue pour simplifier et accélérer le processus de création de modèles prédictifs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Souleymane BA",
    author_email="julesba97@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=get_requirements(requirements_path),
    python_requires=">=3.9",
    package_dir={"": "autolearn"}, 
    packages=find_packages(where="src")
)