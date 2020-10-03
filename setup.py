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
        'torch~=1.6.0',
        'torchvision~=0.7.0',
        'numpy~=1.19.2',
        'matplotlib',
        'tqdm',
        'tensorflow~=1.15',
    ],
    python_requires='>=3.6',
)
