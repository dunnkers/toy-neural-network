#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='toynn',
    version='0.1.0',
    description='Toy Neural Network',
    author='Jeroen Overschie',
    url='https://github.com/dunnkers/toy-neural-network',
    packages=find_packages(include=['toynn', 'toynn.*'])
)