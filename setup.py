from setuptools import setup, find_packages

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

short_description = """A ground heat exchanger design tool with the advanced and
unmatched capability of automatic borehole field selection
based on drilling geometric land constraints."""

setup(
    name='ghedesigner',
    install_requires=[
        'pygfunction @ git+https://github.com/MassimoCimmino/pygfunction@5465044309c1193514f64574680cd430249aef29',
        'wheel',  # I believe once we are installing pygfunction from wheels, we don't need this line anymore
        'numpy>=1.19.2',
        'scipy>=1.6.2',
        'opencv-python==4.5.4.58'
    ],
    url='https://github.com/BETSRG/GHEDTOSU',
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.2',
    packages=find_packages(),
    include_package_data=True,
    author='Jack C. Cook',
    author_email='jack.cook@okstate.edu',
    entry_points={
        'console_scripts': ['ghedesigner=ghedesigner.utilities:dummy_entry_point']
    }
)
