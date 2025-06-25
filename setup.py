"""Setup configuration for MADWE project"""

from setuptools import setup, find_packages

setup(
    name="madwe",
    version="0.1.0",
    description="Multi-Agent Diffusion World Engine for Real-Time Game Content Generation",
    author="Ankit Gole & Shreya Boyane",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies loaded from requirements.txt
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "black>=24.10.0",
            "flake8>=7.1.1",
            "mypy>=1.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "madwe-train=models.diffusion.lora_trainer:main",
            "madwe-benchmark=scripts.benchmark:main",
        ],
    },
)
