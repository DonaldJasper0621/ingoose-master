import setuptools

setuptools.setup(
    name="igoose",
    version="0.0.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "nemo_toolkit[all]==1.19.0",
        "praat-parselmouth==0.4.3",
    ],
)
