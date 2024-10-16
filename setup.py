from setuptools import setup, find_packages

setup(
    name='green_score',
    version='0.0.8',
    author='Sophie Ostmeier, Jean-Benoit Delbrouck',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch==2.2.2',
        'transformers==4.40.0',
        'accelerate==0.30.1',
        'pillow==10.3.0',
        'sentencepiece==0.2.0',
        'sentence-transformers==3.0.1',
        'datasets==2.19.0',
        'torchvision==0.17.2',
        'opencv-python==4.10.0.84',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'pandas'
    ],
    python_requires='>=3.8,<3.9',
    packages=find_packages(),
    zip_safe=False
)