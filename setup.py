from setuptools import setup, Extension, find_packages
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
    name="mvcrender",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    python_requires=">=3.13",
    install_requires=["numpy>=2.1"],
)
