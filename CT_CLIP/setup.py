from setuptools import setup, find_packages

setup(
  name = 'ct-clip',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.1',
  description = 'CT-CLIP',
  install_requires=[
    'beartype',
    'einops>=0.6',
    'ftfy',
    'regex',
    #'torch==2.0.1',
    'torchvision',
    "XlsxWriter",
    "h5py",
    "matplotlib",
    "seaborn",
    "wilds",
    'ImageNetV2_pytorch @ git+https://github.com/modestyachts/ImageNetV2_pytorch.git',
    "click",
    "appdirs",
    "attr",
    "nltk"
      ],
)
