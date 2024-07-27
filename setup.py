"""Setuptool-based build for en_dtypes."""

import fnmatch
import platform
import numpy as np
from setuptools import Extension
from setuptools import setup
from setuptools.command.build_py import build_py as build_py_orig

if platform.system() == "Windows":
  COMPILE_ARGS = [
      "/std:c++17",
      "/DEIGEN_MPL2_ONLY",
      "/EHsc",
      "/bigobj",
  ]
else:
  COMPILE_ARGS = [
      "-std=c++17",
      "-DEIGEN_MPL2_ONLY",
      "-fvisibility=hidden",
      # -ftrapping-math is necessary because NumPy looks at floating point
      # exception state to determine whether to emit, e.g., invalid value
      # warnings. Without this setting, on Mac ARM we see spurious "invalid
      # value" warnings when running the tests.
      "-ftrapping-math",
  ]

exclude = ["third_party*"]


class build_py(build_py_orig):  # pylint: disable=invalid-name

  def find_package_modules(self, package, package_dir):
    modules = super().find_package_modules(package, package_dir)
    return [  # pylint: disable=g-complex-comprehension
        (pkg, mod, file)
        for (pkg, mod, file) in modules
        if not any(
            fnmatch.fnmatchcase(pkg + "." + mod, pat=pattern)
            for pattern in exclude
        )
    ]


setup(
    ext_modules=[
        Extension(
            "en_dtypes._en_dtypes_ext",
            [
                "en_dtypes/src/dtypes.cc",
                "en_dtypes/src/numpy.cc",
            ],
            include_dirs=[
                "third_party/eigen",
                "en_dtypes",
                np.get_include(),
            ],
            extra_compile_args=COMPILE_ARGS,
        )
    ],
    cmdclass={"build_py": build_py},
    author='dengguojie',
)
