from setuptools import setup, find_packages

setup(
    name="green_score",
    version="0.1.0",
    author="Sophie Ostmeier, Jean-Benoit Delbrouck",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # Floors, not pins — this is a library, so users need room to
        # resolve against their own environments.
        "torch>=2.6",
        "torchvision>=0.21",       # first version with cp313 wheels
        # transformers held below v5: v5 changes the dtype default to
        # "auto", which can silently shift scores. Separate decision.
        "transformers>=4.46,<5",
        "numpy>=1.24",
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
        "hf_transfer",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*"]),
    zip_safe=False,
)
