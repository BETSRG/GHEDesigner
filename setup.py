# Jack C. Cook
# Tuesday, September 28, 2021

from setuptools import setup, find_packages
import subprocess
import sys
import os

try:
    import git
except ModuleNotFoundError:
    subprocess.call([sys.executable, '-m', 'pip', 'install', 'gitpython'])
    import git


def getreqs(fname):
    """
    Get the requirements list from the text file
    JCC 03.10.2020
    :param fname: the name of the requirements text file
    :return: a list of requirements
    """
    file = open(fname)
    data = file.readlines()
    file.close()
    return [data[i].replace('\n', '') for i in range(len(data))]


def pull_first():
    """This script is in a git directory that can be pulled."""
    cwd = os.getcwd()
    gitdir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(gitdir)
    g = git.cmd.Git(gitdir)
    try:
        # g.execute(['git', 'lfs', 'pull'])  # this is for the git-lfs tracked files
        g.execute(['git', 'submodule', 'update', '--init'])  # this is to pull in the submodule(s)
    except git.exc.GitCommandError:
        raise RuntimeError("Make sure git-lfs is installed!")
    os.chdir(cwd)

pull_first()

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='GLHEDT',
      install_requires=['matplotlib>=3.3.4',
                        'numpy>=1.19.2',
                        'Pillow>=8.1.0',
                        'scipy>=1.6.2',
                        'pandas>=1.3.2',
                        'natsort>=7.1.1',
                        'gFunctionDatabase>=0.3',
                        'openpyxl>=3.0.8'],
      url='https://github.com/j-c-cook/GLHEDT',
      download_url='https://github.com/j-c-cook/GLHEDT/archive/v0.1.tar.gz',
      long_description=long_description,
      long_description_content_type='text/markdown',
      version='0.1',
      packages=find_packages(),
      include_package_data=True,
      author='Jack C. Cook',
      author_email='jack.cook@okstate.edu',
      description='A ground loop heat exchanger design tool.')
