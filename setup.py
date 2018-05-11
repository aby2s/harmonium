from setuptools import setup
from setuptools import find_packages


setup(name='harmonium',
      version='0.0.2',
      description='Framework for building RBM',
      author='Nikolay Zenovkin',
      author_email='aby2sz@gmail.com',
      url='https://github.com/aby2s/harmonium',
      download_url='https://github.com/aby2s/harmonium',
      license='MIT',
      install_requires=['tensorflow','numpy', 'Pillow', 'scikit-learn'],
      packages=find_packages())