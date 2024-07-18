from setuptools import setup, find_packages

setup(
    name='pydca',
    version='0.1.0',
    packages=find_packages(),
    description='This package contains classes and functions for implementing direct cascaded machine learning models described by Littmann and Ritter.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Heiko Donat',
    author_email='info@hdonat.net',
    url='https://github.com/yourusername/my_package',
    license='MIT',
    install_requires=[
        # List of dependencies
        'numpy',
        'pandas',
        'torch',
        'scikit-learn',
        'scikit-neuralnetwork',
        'jax',
        'jaxlib'
        # etc.
    ],
    classifiers=[
        # Trove classifiers
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)