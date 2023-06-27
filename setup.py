from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import os
import sys

if sys.platform.startswith('win'):
    extra_compile_args = ['/std:c++20', '/O2']
else:
    extra_compile_args = ['-std=c++20', '-Wno-deprecated', '-O3']
# os.environ['CC'] = 'g++'
# os.environ['CXX'] = 'g++'

extensions = [
    Extension(
        "pyqumo.simulations.networks.simulator", [
            "pyqumo/simulations/networks/simulator.pyx",
            "pyqumo/_impl/functions.cpp",
            "pyqumo/randoms/_impl/randoms.cpp",
            "pyqumo/simulations/networks/_impl/networks/base.cpp",
            "pyqumo/simulations/networks/_impl/networks/components.cpp",
            "pyqumo/simulations/networks/_impl/networks/journals.cpp",
            "pyqumo/simulations/networks/_impl/networks/simulation.cpp",
            "pyqumo/simulations/networks/_impl/networks/statistics.cpp",
            "pyqumo/simulations/networks/_impl/networks/system.cpp",
            "pyqumo/simulations/networks/_impl/networks/marshal.cpp",
        ],
        include_dirs=[
            'pyqumo/_impl',
            'pyqumo/randoms/_impl',
            'pyqumo/simulations/networks/_impl/networks',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        # extra_compile_args=["-std=c++20", "-Wno-deprecated", "-O3"],
        extra_link_args=["-std=c++20"]
    ),
    Extension(
        "pyqumo.randoms.variables", [
            "pyqumo/randoms/variables.pyx",
            "pyqumo/_impl/functions.cpp",
            "pyqumo/randoms/_impl/randoms.cpp"
        ],
        include_dirs=[
            'pyqumo/_impl',
            'pyqumo/randoms/_impl',
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
        # extra_compile_args=["-std=c++20", "-Wno-deprecated", "-O3"],
        extra_link_args=["-std=c++20"]
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
