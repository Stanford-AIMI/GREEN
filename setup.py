from setuptools import setup, find_packages

setup(
    name='green_score',
    version='0.0.5',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
        'transformers',
        'accelerate',
        'Pillow',
        'matplotlib',
        'sentencepiece',
        'protobuf',
        'sentence_transformers'
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    zip_safe=False)
