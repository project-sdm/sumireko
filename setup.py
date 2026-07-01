import subprocess
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

MODULE_NAME = "shared.text._spimi_cpp"

ext_modules = [
    Pybind11Extension(
        MODULE_NAME,
        [
            "spimi_cpp/main.cpp",
        ],
        cxx_std=20,
    ),
]


class CustomBuildExt(build_ext):
    def run(self):
        super().run()

        output_dir = Path(".") if self.inplace else Path(self.build_lib)
        subprocess.run(
            [sys.executable, "-m", "pybind11_stubgen", MODULE_NAME, "-o", str(output_dir)],
            check=True,
        )


_ = setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
)
