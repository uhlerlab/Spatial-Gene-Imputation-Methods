from setuptools import setup, find_packages

setup(
    name="spagim",
    version="1.0.2",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "scanpy",
        "scvi-tools",
        "matplotlib",
        "scipy",
        "anndata",
        'tangram-sc',
        'scenvi'
    ],
    author="Kai Cao",
    author_email="caokai1073@gmail.com",
    description="Revealing the biological utility of gene imputation in spatial transcriptomics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/uhlerlab/Spatial-Gene-Imputation-Methods", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)