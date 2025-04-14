from setuptools import setup, find_packages

setup(
    name='gppyro_emulator',
    version='0.1.0',
    description='A package for GP emulation and Bayesian inference with Pyro.',
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    packages=find_packages(),
    install_requires=[
        'torch>=1.10',
        'pyro-ppl>=1.8',
        'gpytorch>=1.6',
        'numpy>=1.20',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose an appropriate license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
) 