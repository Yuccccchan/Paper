from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tsn-wifi-scheduler",
    version="1.0.0",
    author="Paper Reproduction",
    description="TSN-WiFi Network Scheduler - Reproduction of 'Pay Attention to Network'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
    ],
)
