from setuptools import setup,find_packages
import sys, os

setup(name="snake-oil-ml",
      description="For slowing down deep learning research",
      version='0.1',
      author='Marc Finzi',
      author_email='maf388@cornell.edu',
      license='MIT',
      python_requires='>=3.7',
      install_requires=['torch>=1.3','torchvision','pandas','numpy','matplotlib','dill','tqdm>=4.40'
      'sklearn','torchcontrib'],
      extra_require = {
            'tb':['tensorboardX']
            }#,'torch-scatter','torch-sparse','torch-cluster','torch-geometric','torch-spline-conv'],
      packages=find_packages(),#["oil",],#find_packages()
      long_description=open('README.md').read(),
)
#pathToThisFile = os.path.dirname(os.path.realpath(__file__))
# add to .bashrc
#sys.path.append(pathToThisFile)
