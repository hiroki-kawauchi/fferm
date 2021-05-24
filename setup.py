# -*- coding: utf-8 -*-
import os, sys
from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

setup(
    name='fferm',
    version='0.0.1',
    description='Face and facial expression recognition from movie',
    long_description=readme,
    author='Hiroki Kawauchi',
    author_email='-',
    url='https://github.com/hiroki-kawauchi/fferm',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=read_requirements(),
    test_suite='tests'
)

