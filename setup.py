from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build_ext to handle NumPy include directories."""

    def finalize_options(self):
        """Delay NumPy import until build time."""
        super().finalize_options()
        # Import numpy here, after it's been installed as a build dependency
        import numpy
        # Add NumPy include directory to all extensions
        for ext in self.extensions:
            ext.include_dirs.append(numpy.get_include())


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
