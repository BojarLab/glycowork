import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glycowork",
    version="1.4.0",
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
    install_requires=["scikit-learn", "regex", "networkx>=3.0",
                      "statsmodels", "scipy", "gdown",
                      "seaborn", "xgboost", "mpld3",
                      "pandas>=1.3", "matplotlib-inline"],
    extras_require={'all':["torch_geometric", "torch", "CairoSVG",
                           "drawSvg~=2.0", "glyles", "pubchempy", "requests",
                           "Pillow", "openpyxl", "py3Dmol", "gdown"],
                    'dev':["torch_geometric", "torch", "CairoSVG",
                           "drawSvg~=2.0", "glyles", "pubchempy", "requests",
                           "Pillow", "openpyxl", "py3Dmol", "gdown", "pytest"],
                    'ml':["torch_geometric", "torch"],
                    'draw':["CairoSVG", "drawSvg~=2.0", "Pillow",
                            "openpyxl"],
                    'chem':["glyles", "pubchempy", "requests", "py3Dmol"]},
)
