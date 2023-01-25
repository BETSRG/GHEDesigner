from os import path

from setuptools import setup

from ghedesigner import VERSION

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

short_description = """A ground heat exchanger design tool with the advanced and
unmatched capability of automatic borehole field selection
based on drilling geometric land constraints."""

setup(
    name='ghedesigner',
    install_requires=[
        'click',
        'pygfunction',
        'numpy>=1.19.2',
        'scipy>=1.6.2',
        'opencv-python==4.5.4.58'
    ],
    url='https://github.com/BETSRG/GHEDTOSU',
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    version=VERSION,
    packages=['ghedesigner'],
    author='Jeffrey D. Spitler,',
    author_email='spitler@okstate.edu',
    entry_points={
        'console_scripts': ['ghedesigner=ghedesigner.manager:run_manager_from_cli']
    }
)
