# Installation instructions

Preferably execute this code inside a virtual environment. This setup was tested on Ubuntu 18.04 and macOS Catalina 10.15.6.

1. Setup a virtual env: `python3 -m venv venv`  
2. Activate the virtual env: `source venv/bin/activate`  
3. Make sure to have recent version of pip: `pip install -U pip`  
4. Install package: `pip install git+https://github.com/yeedHub/neurocatChallenge.git`

**Note:** pytorch for Windows must be installed as mentioned on their website. For example 
pytorch without CUDA:   
`pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html`


Inside this virtual environment, after successful installation, you can now use
puzzle.py (for example execute a single test run with img.jpg `python test_puzzle.py`),
and run the jupyter notebook `report.ipynb`.
