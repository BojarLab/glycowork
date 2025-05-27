import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glycowork",
    version="1.6.0",
    author="Daniel Bojar",
    author_email="daniel.bojar@gu.se",
    description="Package for processing and analyzing glycans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BojarLab/glycowork",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv', '*.pkl', '*.jpg', '*.pt', '*.json']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=["numpy", "matplotlib", "scikit-learn", "networkx>=3.0",
                      "statsmodels", "scipy", "seaborn", "xgboost", "bokeh",
                      "pandas>=1.3", "setuptools>=64.0", "IPython",
                      "huggingface_hub>=0.16.0", "drawSvg~=2.0", "Pillow",
                            "openpyxl", "glycorender>=0.1.5"],
    extras_require={'all':["torch_geometric", "torch",
                           "glyles", "pubchempy", "requests", "py3Dmol"],
                    'dev':["torch_geometric", "torch",
                           "glyles", "pubchempy", "requests", "py3Dmol", "pytest"],
                    'ml':["torch_geometric", "torch"],
                    'chem':["glyles", "pubchempy", "requests", "py3Dmol"]},
)
