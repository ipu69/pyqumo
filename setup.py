from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import sys

if sys.platform.startswith('win'):
    extra_compile_args = ['/std:c++14', '/O2']
else:
    extra_compile_args = ['-std=c++14', '-Wno-deprecated', '-O3']

extensions = [
    Extension(
        "pyqumo.cqumo.variables", [
            "pyqumo/cqumo/variables.pyx",

            "cqumo/cqumo/randoms.cpp",
            "cqumo/cqumo/utils/functions.cpp",
        ],
        include_dirs=[
            'cqumo',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=["-std=c++14"]
    ),
    Extension(
        "pyqumo.cqumo.models.oqnet.model", [
            "pyqumo/cqumo/models/oqnet/model.pyx",

            "cqumo/cqumo/randoms.cpp",
            "cqumo/cqumo/utils/functions.cpp",
            "cqumo/cqumo/statistics/series.cpp",
            "cqumo/cqumo/statistics/statistics.cpp",
            "cqumo/cqumo/models/oqnet/components.cpp",
            "cqumo/cqumo/models/oqnet/journals.cpp",
            "cqumo/cqumo/models/oqnet/marshal.cpp",
            "cqumo/cqumo/models/oqnet/simulation.cpp",
            "cqumo/cqumo/models/oqnet/system.cpp",
        ],
        include_dirs=[
            'cqumo',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=["-std=c++14"]
    ),
]

compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
}


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='pyqumo',
    version='1.1.0',
    packages=find_packages(exclude=["tests"]),
    description='Queueing Models in Python',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
    ],
    keywords='queueing systems, markov chains',
    url='https://github.com/ipu69/pyqumo',
    author='Andrey Larionov',
    author_email='larioandr@gmail.com',
    license='MIT',
    scripts=[],
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'scipy',
        'tabulate',
        'cython',
        'pebble',
        'click',
        'pandas',
    ],
    include_package_data=True,
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives
    ),
    extras_require={
        "docs": ["sphinx", "sphinx-rtd-theme"]
    }
)
