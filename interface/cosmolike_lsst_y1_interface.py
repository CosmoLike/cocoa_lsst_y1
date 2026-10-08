"""Loads cosmolike_lsst_y1_interface.so if Python ever imports this stub instead.

The module cosmolike_lsst_y1_interface is the shared library
cosmolike_lsst_y1_interface.so in this directory, built from interface.cpp
by MakefileCosmolike. When both files sit in the same directory, Python's
import system tries extension modules (.so) before source files (.py), so
this stub does not run. It is the stub that setuptools (bdist_egg) writes
next to a compiled extension: __bootstrap__ finds the .so next to this file
and loads it in place of this module (imp.load_dynamic). The imp module
exists up to Python 3.11 and was removed in Python 3.12.
"""
def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'cosmolike_lsst_y1_interface.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
