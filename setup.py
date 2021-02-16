import setuptools

setuptools.setup(
    name="Neurocat-Challenge-Yigit-Akcay",
    version="1.0",
    author="Yigit Akcay",
    author_email="",
    description="Solution for the neurocat challenge.",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'torch~=1.6.0',
        'torchvision~=0.7.0',
        'numpy==1.18.5',
        'matplotlib~=3.2.1',
        'tqdm~=4.48.0',
        'tensorflow~=1.15',
        'Pillow==7.2.0',
        'jupyter~=1.0.0',
        'ipykernel~=5.3.4',
        'ipywidgets~=7.5.1'
    ],
    python_requires='>=3.6',
)
