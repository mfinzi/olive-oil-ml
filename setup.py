from setuptools import setup,find_packages
import sys, os

setup(name="olive-oil-ml",
      description="For slowing down deep learning research",
      version='0.1',
      author='Marc Finzi',
      author_email='maf388@cornell.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['torch>=1.2','torchvision','pandas','numpy','matplotlib','dill','tqdm>=4.38','natsort','sklearn','torchcontrib'],
      extras_require = {
            'TBX':['tensorboardX']
            },
      packages=find_packages(),#["oil",],#find_packages()
      long_description=open('README.md').read(),
)
#pathToThisFile = os.path.dirname(os.path.realpath(__file__))
# add to .bashrc
#sys.path.append(pathToThisFile)
