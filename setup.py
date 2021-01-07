import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="glycowork-Bribak", # Replace with your own username
    version="0.0.1",
    author="Daniel Bojar",
    author_email="daniel@bojar.net",
    description="Package containing helper functions for processing and analysis of glycans",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BojarLab/glycowork",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
