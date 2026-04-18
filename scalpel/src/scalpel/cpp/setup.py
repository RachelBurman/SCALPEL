"""
Build script for the SCALPEL C++ chunker extension.

Run from the scalpel/ directory:
    python src/scalpel/cpp/setup.py build_ext --inplace

The compiled extension (_scalpel_chunker.pyd on Windows) is placed in
src/scalpel/cpp/ and imported automatically by the Python chunker.
"""

from pathlib import Path
from setuptools import setup, Extension
import pybind11

cpp_dir = Path(__file__).parent

ext = Extension(
    name="_scalpel_chunker",
    sources=[
        str(cpp_dir / "chunker.cpp"),
        str(cpp_dir / "bindings.cpp"),
    ],
    include_dirs=[
        str(cpp_dir),
        pybind11.get_include(),
    ],
    language="c++",
    extra_compile_args=["/std:c++17", "/O2", "/W3"] if __import__("sys").platform == "win32"
                       else ["-std=c++17", "-O3", "-Wall"],
)

setup(
    name="scalpel_chunker",
    version="1.0.0",
    ext_modules=[ext],
)
