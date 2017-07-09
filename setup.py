from setuptools import setup
from setuptools import find_packages


setup(name='boltzmann',
      version='0.0.1',
      description='Framework for building RBM',
      author='Nikolay Zenovkin',
      author_email='aby2sz@gmail.com',
      url='https://github.com/aby2s/boltzmann',
      download_url='https://github.com/aby2s/boltzmann',
      license='MIT',
      install_requires=['tensorflow','numpy', 'Pillow', 'scikit-learn'],
      packages=find_packages())