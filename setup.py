from pathlib import Path

from setuptools import setup

from ghedesigner import VERSION

readme_file = Path(__file__).parent.resolve() / 'README.md'
readme_contents = readme_file.read_text(encoding='utf8')

short_description = """A ground heat exchanger design tool with the capability
to select and size flexibly configured borehole fields that are customized
for specific building and property constraints."""

setup(
    name='GHEDesigner',
    install_requires=[
        'click>=8.1.3',
        'jsonschema>=4.17.3',
        'numpy>=1.24.2',
        'opencv-python==4.7.0.68',
        'pygfunction>=2.2.2',
        'scipy>=1.10.0'
    ],
    url='https://github.com/BETSRG/GHEDesigner',
    description=short_description,
    license='BSD-3',
    long_description=readme_contents,
    long_description_content_type='text/markdown',
    version=VERSION,
    packages=['ghedesigner'],
    include_package_data=True,
    package_data={'ghedesigner': ['schemas/*.json']},
    author='Jeffrey D. Spitler',
    author_email='spitler@okstate.edu',
    entry_points={
        'console_scripts': ['ghedesigner=ghedesigner.manager:run_manager_from_cli']
    },
    python_requires='>=3.8',
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10'
    ]
)
