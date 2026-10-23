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
import os
import sys
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

class CMakeBuildExt(build_ext):
    """
    Custom build class that intercepts the build process to run CMake 
    inside the 'charging' subdirectory before compiling the Cython extension.
    """
    def run(self):
        # 1. Path setups
        cwd = os.getcwd()
        charging_dir = os.path.abspath("charging")
        build_dir = os.path.join(charging_dir, "cmake_build")
        lib_dir = os.path.join(charging_dir, "lib")
        
        # 2. Ensure directories exist
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(lib_dir, exist_ok=True)
        
        # 3. Run CMake and Make natively inside charging/cmake_build
        try:
            print("--- Running CMake configuration inside charging/ ---")
            subprocess.check_call(["cmake", ".."], cwd=build_dir)
            
            print("--- Running Make compilation ---")
            subprocess.check_call(["make"], cwd=build_dir)
            
            # 4. Safely move the output binary to where setup.py expects it
            built_lib = os.path.join(build_dir, "libcharging.so")
            target_lib = os.path.join(lib_dir, "libcharging.so")
            if os.path.exists(built_lib):
                print(f"--- Copying compiled library to {target_lib} ---")
                import shutil
                shutil.copy(built_lib, target_lib)
                
        except subprocess.CalledProcessError as e:
            print(f"CMake/Make compilation failed: {e}")
            sys.exit(1)
            
        # 5. Fall back to standard Cython/Setuptools build execution
        super().run()

# Define Extension 1
ext_coagulatio = Extension(
    "coagulatio",
    ["coagulatio/coagulatio.pyx"],
    extra_compile_args=["-Ofast", "-fopenmp"],
    extra_link_args=['-fopenmp']
)

# Define Extension 2
ext_charging = Extension(
    "charging", 
    ["charging/charging.pyx"],
    include_dirs=[
        "charging/include/",
        "charging/external/liblsoda/src/",
        numpy.get_include()
    ],
    libraries=["charging", "lsoda", "m"],
    library_dirs=[
        "charging/lib/",
        "charging/external/liblsoda/src/"
    ],
    extra_compile_args=["-Ofast", "-fopenmp"],
    extra_link_args=[
        "-fopenmp",
        "-Wl,-rpath=$ORIGIN/charging/lib/",
        "-Wl,-rpath=$ORIGIN/charging/external/liblsoda/src/"
    ]
)

extensions = [ext_coagulatio, ext_charging]

setup(
    name="coagulatio-charging-package",
    version="1.0",
    # Pass our custom subclass here instead of the default build_ext
    cmdclass={"build_ext": CMakeBuildExt},
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[numpy.get_include()]
)
