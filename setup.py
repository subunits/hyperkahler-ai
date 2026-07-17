"""Setup script for hyperkahler-ai package."""

from setuptools import setup, find_packages

setup(
    name="hyperkahler-ai",
    version="0.1.0",
    description="Unified Hyperkähler AI framework: quaternion autoencoders + transformers",
    author="Subunits",
    author_email="contact@subunits.dev",
    url="https://github.com/subunits/hyperkahler-ai",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "matplotlib>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
