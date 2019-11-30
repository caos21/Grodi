#
#   Copyright 2019 Benjamin Santos
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
""" setup.py, compiles coagulatio and charging extensions
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import numpy

EXT_COAGULATIO = [Extension("coagulatio",
                            ["coagulatio/coagulatio.pyx"],
                            extra_compile_args=["-Ofast", "-fopenmp"],
                            extra_link_args=['-fopenmp'])]


EXT_CHARGING = [Extension("charging", ["charging/charging.pyx"],
                          include_dirs=["charging/include/",
                                        "charging/external/liblsoda/src/",
                                        numpy.get_include()],
                          libraries=["charging", "lsoda", "m"],
                          library_dirs=["charging/lib/",
                                        "charging/external/liblsoda/src/"],
                          extra_compile_args=["-Ofast", "-fopenmp"],
                          extra_link_args=["-fopenmp",
                                           "-Wl,-rpath=charging/lib/",
                                           "-Wl,-rpath=charging/external/liblsoda/src/"])]

setup(
    name="coagulatio",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(EXT_COAGULATIO, annotate=True, ),
    include_dirs=[numpy.get_include()])

setup(
    name="charging",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(EXT_CHARGING, annotate=True, ),
)
