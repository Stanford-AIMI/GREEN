from setuptools import setup, find_packages

setup(
    name="green_score",
    version="0.0.13",
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
        # STEP 1: torch unpinned from 2.2.2 → 2.4/2.5 window.
        # Ceiling at <2.6 is deliberate: torch 2.6 flipped torch.load to
        # weights_only=True, which breaks transformers 4.40 checkpoint loading.
        # Lift this ceiling in step 2, together with the transformers bump.
        "torch>=2.4,<2.6",
        "torchvision>=0.19,<0.21",
        # still held
        "transformers==4.40.0",
        "numpy<2",
        # loosened previously
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
