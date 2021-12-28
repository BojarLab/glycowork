import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glycowork",
    version="0.3.0",
    author="Daniel Bojar",
    author_email="daniel.bojar@gu.se",
    description="Package containing helper functions for processing and analysis of glycans",
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
    python_requires='>=3.6',
    install_requires=["sklearn", "regex", "networkx",
                      "statsmodels", "scipy", "torch",
                      "seaborn", "xgboost", "mpld3",
                      "requests"],
)
