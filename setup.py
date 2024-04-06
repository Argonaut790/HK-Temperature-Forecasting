from setuptools import setup, find_packages

NAME = 'HK-Temperature-Forecasting'
DESCRIPTION = ''
AUTHOR = 'Billy Tse'
EMAIL = 'tung23966373@gmail.com'
VERSION = '1.0.0'
LICENSE = 'MIT'
KEYWORDS = ['']
REQUIREMENTS = ['loguru', 'numpy', 'matplotlib', 'scipy', 'mysql-connector',
                'scikit-learn', 'polars', 'xgboost', 'ipython', 'ipykernel', 'seaborn',
                'lightgbm', 'tensorflow', 'keras', 'skforecast']

setup(name=NAME, version=VERSION, packages=find_packages(), 
      install_requires=REQUIREMENTS, author=AUTHOR, author_email=EMAIL,
      description=DESCRIPTION, license=LICENSE, keywords=KEYWORDS)

# pip3 freeze > requirements.txt