import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glycowork",
    version="0.8.0",
    author="Daniel Bojar",
    author_email="daniel.bojar@gu.se",
    description="Package for processing and analyzing glycans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BojarLab/glycowork",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv', '*.pkl', '*.jpg', '*.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=["scikit-learn", "regex", "networkx>=3.0",
                      "statsmodels", "scipy", "torch",
                      "seaborn", "xgboost", "mpld3",
                      "requests", "pandas>=1.3", "glyles",
                      "pubchempy", "matplotlib-inline",
                      "python-louvain"],
    extras_require={'all':["torch_geometric", "pycairo", "CairoSVG",
                           "drawSvg~=2.0"],
                    'dev':["torch_geometric", "pycairo", "CairoSVG",
                           "drawSvg~=2.0"],
                    'ml':["torch_geometric"],
                    'draw':["pycairo", "CairoSVG", "drawSvg~=2.0"]},
)
