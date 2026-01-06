"""Setup script for Agentic Organ Generation."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentic-organ-generation",
    version="0.1.0",
    author="Erick Gross",
    author_email="erickgross1924@gmail.com",
    description="LLM/agent-driven generation and validation of 3D organ structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ErickGross-19/Agentic-Organ-Generation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "trimesh>=3.10.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "mesh": ["pymeshfix>=0.16.0"],
        "llm": ["openai>=1.0.0", "anthropic>=0.18.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        "all": [
            "pymeshfix>=0.16.0",
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agentic-organ=automation.cli:main",
        ],
    },
)
