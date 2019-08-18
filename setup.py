#!usr/bin/python3

"""AQTLib: Asynchronous Quantitative Trading Library
(https://github.com/kelvingao/aqtlib)
"""

import os
from setuptools import setup
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))

__version__ = None
exec(open(os.path.join(here, 'aqtlib', 'version.py')).read())

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aqtlib',
    version=__version__,
    description='Asynchronous Quantitative Trading Library',
    long_description=long_description,
    url='https://github.com/kelvingao/aqtlib',
    author='Kelvin Gao',
    author_email='89156201@qq.com',
    license='MIT',
    python_requires='>=3.6',
    classifiers=[
      # Full list: https://pypi.org/pypi?%3Aaction=list_classifiers
      'Development Status :: 1 - Planning',
      'Intended Audience :: Developers',
      'Topic :: Office/Business :: Financial :: Investment',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
    ],
    platforms=['any'],
    keywords='aqtlib algo trading interactive brokers tws asyncio',
    packages=['aqtlib'],
    install_requires=['ib_insync>=0.9.56'],
)
