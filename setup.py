from setuptools import setup, find_packages

setup(
  name = 'liptrf',
  packages = find_packages(exclude=['examples']),
  version = '0.1.0',
  license='MIT',
  description = 'Lipschitz training of transformers - Pytorch',
  author = 'Sagar Verma and Kavya Gupta',
  author_email = 'sagar@granular.ai',
  url = 'https://github.com/sagarverma/transformer-lipschitz',
  keywords = [
    'artificial intelligence',
    'transformers',
    'image recognition'
  ],
  install_requires=[
    'einops>=0.4.1',
    'torch>=1.10',
    'torchvision',
    'jupyterlab',
    'matplotlib',
    'timm==0.5.4',
    'scipy==1.8.0',
    'webdataset',
    'tqdm'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8.5',
  ],
)