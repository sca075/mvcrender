from setuptools import setup, Extension
import numpy

ext_modules = [
    Extension(
        "mvcrender.autocrop._native",
        sources=["src/mvcrender/autocrop/_native.c"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        language="c",
    )
]

setup(
    ext_modules=ext_modules,
)
