from setuptools import setup, find_packages

setup(
    name="green_score",
    version="0.0.12",
    author="Sophie Ostmeier, Jean-Benoit Delbrouck",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # held for now — loosen one at a time after this passes
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.40.0",
        "numpy<2",
        # loosened
        "accelerate>=0.30",
        "pillow>=10.3",
        "sentencepiece>=0.2",
        "sentence-transformers>=3.0",
        "datasets>=3.2",
        "opencv-python>=4.10",
        "dill>=0.3.8",
        "protobuf>=5.29",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "pandas",
        "pytest",
        "hf_transfer"
    ],
    python_requires=">=3.9,<3.13",
    packages=find_packages(),
    zip_safe=False,
)
