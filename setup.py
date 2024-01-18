from setuptools import setup, find_packages

NAME = 'argonaut-temp-forecast'
DESCRIPTION = ''
AUTHOR = 'Billy Tse'
EMAIL = 'tung23966373@gmail.com'
VERSION = '1.0.0'
LICENSE = 'MIT'
KEYWORDS = ['']
REQUIREMENTS = ['loguru', 'numpy', 'matplotlib', 'scipy', 'mysql-connector',
                'scikit-learn', 'pandas', 'xgboost', 'ipython', 'seaborn', 
                'torch', 'torchvision', 'torchaudio', 'ipykernel']

setup(name=NAME, version=VERSION, packages=find_packages(), 
      install_requires=REQUIREMENTS, author=AUTHOR, author_email=EMAIL,
      description=DESCRIPTION, license=LICENSE, keywords=KEYWORDS)

# pip3 freeze > requirements.txt