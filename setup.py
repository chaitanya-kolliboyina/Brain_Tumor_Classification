from setuptools import setup, find_packages

setup(
    name="Brain_tumor_classification",
    version="0.1.0",
    packages = find_packages(),
    install_requires = [
        "torch",
        "transformers",
        "torchvision",
        "numpy",
        "matplotlib",
        "seaborn",
        "albumentations"
    ],
    description="Brain tumor classification using transformers on AMD GPU using ROCm",
    author="Chaitanya KVS",
    author_email="kvschaitanya17@gmail.com",
    url="https://github.com/chaitanya-kolliboyina/Brain_Tumor_Classification",
    classifiers=[                       # Metadata for categorizing the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",  
)