from setuptools import setup, find_packages

setup(
    name='SSLEmbeddingNorms',
    version='0.1',
    license='MIT',
    packages=[
        'CLA', # CLA stands for Contrastive Learning Analysis
        'CLA.experiments',
        'CLA.training',
        'CLA.utils'
    ],
)
