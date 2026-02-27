import importlib.util
import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def _get_numpy_include():
    """Return the numpy include path without requiring C extensions to load.

    On restricted environments (e.g. aarch64-linux-musl sandbox) numpy's .so
    files cannot be loaded, so a direct ``import numpy`` fails.  We fall back
    to locating the headers via importlib, which only inspects the file-system
    and never executes numpy's __init__.py.
    """
    try:
        import numpy
        return numpy.get_include()
    except ImportError:
        pass
    spec = importlib.util.find_spec("numpy")
    if spec and spec.origin:
        numpy_dir = os.path.dirname(spec.origin)
        for candidate in ("_core/include", "core/include"):
            path = os.path.join(numpy_dir, candidate)
            if os.path.isdir(path):
                return path
    return None


class BuildExt(build_ext):
    """Custom build_ext to handle NumPy include directories."""

    def finalize_options(self):
        """Delay NumPy include resolution until build time."""
        super().finalize_options()
        numpy_include = _get_numpy_include()
        if numpy_include:
            for ext in self.extensions:
                ext.include_dirs.append(numpy_include)


ext_modules = [
    Extension(
        "mvcrender.autocrop._native",
        sources=["src/mvcrender/autocrop/_native.c"],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
    Extension(
        "mvcrender.blend._native",
        sources=["src/mvcrender/blend/_native.c"],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
    Extension(
        "mvcrender.draw._native",
        sources=["src/mvcrender/draw/_native.c"],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
    Extension(
        "mvcrender.rooms._native",
        sources=["src/mvcrender/rooms/_native.c"],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
    Extension(
        "mvcrender.material._native",
        sources=["src/mvcrender/material/_native.c"],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
