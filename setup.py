import subprocess
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "spimi_cpp",
        [
            "spimi_cpp/main.cpp",
        ],
        cxx_std=20,
    ),
]


class CustomBuildExt(build_ext):
    def build_extension(self, ext):
        super().build_extension(ext)
        subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", ext.name, "-o", "."],
            check=True,
        )


_ = setup(
    name="spimi_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.14",
)
