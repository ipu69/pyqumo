from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import sys

if sys.platform.startswith('win'):
    extra_compile_args = ['/std:c++14', '/O2']
else:
    extra_compile_args = ['-std=c++14', '-Wno-deprecated', '-O3']
# os.environ['CC'] = 'g++'
# os.environ['CXX'] = 'g++'

extensions = [
    Extension(
        "pyqumo.simulations.networks.model", [
            "pyqumo/simulations/networks/model.pyx",
            "cqumo/core/src/cqumo/functions.cpp",
            "cqumo/core/src/cqumo/statistics.cpp",
            "cqumo/core/src/cqumo/randoms.cpp",
            "cqumo/oqnet/src/cqumo/oqnet/components.cpp",
            "cqumo/oqnet/src/cqumo/oqnet/journals.cpp",
            "cqumo/oqnet/src/cqumo/oqnet/simulation.cpp",
            "cqumo/oqnet/src/cqumo/oqnet/system.cpp",
            "cqumo/oqnet/src/cqumo/oqnet/marshal.cpp",
        ],
        include_dirs=[
            'cqumo/core/src',
            'cqumo/core/src/cqumo',
            'cqumo/core/include',
            'cqumo/oqnet/src/cqumo/oqnet',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        # extra_compile_args=["-std=c++20", "-Wno-deprecated", "-O3"],
        extra_link_args=["-std=c++14"]
    ),
    Extension(
        "pyqumo.randoms.variables", [
            "pyqumo/randoms/variables.pyx",
            "cqumo/core/src/cqumo/functions.cpp",
            "cqumo/core/src/cqumo/randoms.cpp"
        ],
        include_dirs=[
            'cqumo/core/src/cqumo',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        # extra_compile_args=["-std=c++20", "-Wno-deprecated", "-O3"],
        extra_link_args=["-std=c++14"]
    )
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
    packages=find_packages(exclude=["tests",]),
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
#     packages=['pyqumo'],
#     py_modules=['pyqumo'],
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
