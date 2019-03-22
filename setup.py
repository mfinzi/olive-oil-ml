from setuptools import setup
import sys, os

setup(name="Pristine-ml",
      description="For slowing down deep learning research",
      version='-0.1',
      author='Marc Finzi',
      author_email='maf388@cornell.edu',
      license='MIT',
      install_requires=['torch','torchvision','pandas','numpy','dill','sklearn'],
      #packages=["."],
)
pathToThisFile = os.path.dirname(os.path.realpath(__file__))
# add to .bashrc
#sys.path.append(pathToThisFile)
