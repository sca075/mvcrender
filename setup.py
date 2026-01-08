from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "mvcrender.autocrop._native",
        sources=["src/mvcrender/autocrop/_native.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    ),
    Extension(
            "mvcrender.blend._native",
            sources=["src/mvcrender/blend/_native.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            language="c",
        ),
    Extension(
            "mvcrender.draw._native",
            sources=["src/mvcrender/draw/_native.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            language="c",
        ),
    Extension(
            "mvcrender.rooms._native",
            sources=["src/mvcrender/rooms/_native.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            language="c",
        ),
]

setup(
    ext_modules=ext_modules,
)
