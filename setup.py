from setuptools import setup, find_packages

setup(
    name="isac-uav",
    version="0.1.0",
    description="ISAC-UAV: Estimation of UAV Parameters Using Monostatic Sensing in ISAC Scenario",
    author="Your Name",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchaudio",
        "tqdm"
    ],
    python_requires=">=3.8",
)
