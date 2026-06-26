from setuptools import setup, find_packages

setup(
    name="fraud-detection-graph-mining",
    version="1.0.0",
    description="Fraud Pattern Detection using Graph Mining Algorithms",
    author="Fraud Detection Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "networkx>=3.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "python-louvain>=0.16",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "gpu": [
            "cudf>=23.0.0",
            "cugraph>=23.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "lightgbm>=4.0.0",
            "catboost>=1.2.0",
            "optuna>=3.3.0",
            "dash>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection=main:main",
        ],
    },
)
