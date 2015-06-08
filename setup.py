#!/usr/bin/python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': '',
    'author': 'Christopher Woodall',
    'url': '',
    'download_url': '',
    'author_email': 'chris@cwoodall.com',
    'version': '0.1',
    'install_requires': ['nose', 'pyaudio', 'matplotlib'],
    'packages': ['pydoppler'],
    'scripts': ['bin/doppler-gestures.py'],
    'name': 'pydoppler',
    'test_suite': 'nose.collector'
}

setup(**config)
