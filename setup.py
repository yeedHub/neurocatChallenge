import setuptools

setuptools.setup(
    name="Neurocat-Challenge-Yigit-Akcay",
    version="1.0",
    author="Yigit Akcay",
    author_email="***REMOVED***",
    description="Solution for the neurocat challenge.",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'tqdm',
        'jupyter'
    ],
    python_requires='>=3.6',
)
