from setuptools import setup, find_packages
setup(
    name='gcd-quotient-faithfulness',
    version='0.1.0',
    author='Peter Farkas',
    description='Generator-Constrained Discovery framework — companion code',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0', 'numpy>=1.24', 'pandas>=2.0',
        'scipy>=1.10', 'scikit-learn>=1.3',
        'matplotlib>=3.7', 'ripser>=0.6', 'openpyxl>=3.1',
    ],
)
