#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re

try:
    from setuptools import setup
    setup
except ImportError:
    from distutils.core import setup
    setup

setup(
    name="charm",
    version='0.1',
    author="Phill Cargile",
    author_email="pcargile@cfa.harvard.edu",
    packages=["charm"],
    url="",
    #license="LICENSE",
    description="Cluster Heirarchical Astrometric Modeling",
    long_description=open("README.md").read() + "\n\n",
    #install_requires=["numpy", "scipy"],
)
